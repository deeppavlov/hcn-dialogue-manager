"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import json
import sys
import copy
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class HybridCodeNetworkModel(object):

    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)

        # initialize session
        self._sess = tf.Session()
        # initialize metric variables
        self.reset_metrics()

        if self.opt.get('pretrained_model'):
            # restore state, parameters and session
            self.restore()
        else:
            sys.stderr.write("<INFO> Initializing model from scratch.\n")
            # initialize parameters
            self.__init_params__()
            # build computational graph
            self.__build__()
            # zero state
            self.reset_state()
            # initialize variables
            self._sess.run(tf.global_variables_initializer())


    def __init_params__(self, params=None):
        params = params or self.opt
        self.learning_rate = params['learning_rate']
        self.n_epoch = params['epoch_num']
        self.n_hidden = params['hidden_dim'] 
        self.n_actions = params['action_size']
        self.obs_size = params['obs_size']

    def __build__(self):
        tf.reset_default_graph()

        # entry points
        self._features = tf.placeholder(tf.float32, [1, self.obs_size], 
                name='features')
        self._state_c = tf.placeholder(tf.float32, [1, self.n_hidden],
                name='state_c') 
        self._state_h = tf.placeholder(tf.float32, [1, self.n_hidden],
                name='state_h') 
        self._action = tf.placeholder(tf.int32, 
                name='ground_truth_action')
        self._action_mask = tf.placeholder(tf.float32, [self.action_size], 
                name='action_mask')

        # input projection
        _Wi = tf.get_variable('Wi', [self.obs_size, self.n_hidden], 
                initializer=xavier_initializer())
        _bi = tf.get_variable('bi', [self.n_hidden], 
                initializer=tf.constant_initializer(0.))

        # add relu/tanh here if necessary
        _projected_features = tf.matmul(self._features, _Wi) + _bi

        _lstm_f = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True)

        _lstm_op, self._state = lstm_f(inputs=_projected_features, 
                state=(self._state_c, self._state_h))

        # reshape LSTM's state tuple (2,128) -> (1,256)
        _state_reshaped = tf.concat(axis=1, values=(self._state.c, self._state.h))

        # output projection
        _Wo = tf.get_variable('Wo', [2*self.n_hidden, self.action_size], 
                initializer=xavier_initializer())
        _bo = tf.get_variable('bo', [self.action_size], 
                initializer=tf.constant_initializer(0.))
        # get logits
        _logits = tf.matmul(_state_reshaped, _Wo) + _bo

        # probabilities normalization : elemwise multiply with action mask
        self._probs = tf.multiply(tf.squeeze(tf.nn.softmax(_logits)), 
                self._action_mask)
        
        self._prediction = tf.arg_max(self._probs, dimension=0)

        self._loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=_logits, labels=self._action)

        self._train_op = tf.train.AdadeltaOptimizer(self.learning_rate)\
                .minimize(self._loss)

        # create collections for easier `restore` operation
        tf.add_to_collection('state', self._state)
        tf.add_to_collection('probs', self._probs)
        tf.add_to_collection('prediction', self._prediction)
        tf.add_to_collection('loss', self._loss)
        tf.add_to_collection('train_op', self._train_op)

    def reset_metrics(self):
        self.n_examples = 0
        self.step = 0
        self.train_loss = 0.
        self.train_acc = 0.
        self.train_f1 = 0.
        self.val_loss = 0.
        self.val_acc = 0.
        self.val_f1 = 0.

    def reset_state(self):
        # set zero state
        self.state_c = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.state_h = np.zeros([1,self.nb_hidden], dtype=np.float32)

    def update(self, features, action, action_mask):
        _, loss_value, self.state_c, self.state_h = self._sess.run(
                [self._train_op, self._loss, self._state.c, self._state.h],
                feed_dict={
                    self._features: features.reshape([1, self.obs_size]),
                    self._action: [action],
                    self._state_c: self.state_c,
                    self._state_h: self.state_h,
                    self._action_mask: action_mask
                    })
        return loss_value

    def predict(self, features, action_mask):
        probs, prediction, self.state_c, self.state_h = self._sess.run(
                [self._probs, self._prediction, self._state.c, self._state.h],
                feed_dict={
                    self._features: features.reshape([1, self.obs_size]),
                    self._state_c: self.state_c,
                    self._state_h: self.state_h,
                    seld._action_mask: action_mask
                    })
        return probs, prediction

    def restore(self, fname=None):
        """Restore graph, important operations and state"""
        fname = fname or self.opt['pretrained_model']
        json_fname, meta_fname = "{}.json".format(fname), "{}.meta".format(fname)
        _ckpt = tf.train.get_checkpoint_state(*os.path.split(fname))

        if _ckpt:
            sys.stderr.write("<INFO> restoring model from '{}'\n".format(meta_fname))
            tf.reset_default_graph()
            #_saver = tf.train.Saver()
            _saver = tf.train.import_meta_graph(meta_fname)
            _saver.restore(self._sess, fname)

            # restore placeholders
            _graph = tf.get_default_graph()
            self._features = _graph.get_tensor_by_name("features:0")
            self._state_c = _graph.get_tensor_by_name("state_c:0")
            self._state_h = _graph.get_tensor_by_name("state_h:0")
            self._action = _graph.get_tensor_by_name("ground_truth_action:0")
            self._action_mask = _graph.get_tensor_by_name("action_mask:0")

            # restore important operations
            self._state = _graph.get_collection('state')[0]
            self._probs = _graph.get_collection('probs')[0]
            self._prediction = _graph.get_collection('prediction')[0]
            self._loss = _graph.get_collection('loss')[0]
            self._train_op = _graph.get_collection('train_op')[0]

            # restore state
            if os.path.isfile(json_fname):
                with open(json_fname, 'r') as f:
                    params, (self.state_c, self.state_h) = json.load(f)
                    self._init_params(params)
            else:
                sys.stderr.write("<ERR>'{}' not found\n".format(json_fname))
                exit()
        else:
            sys.stderr.write("<ERR> checkpoint '{}' not found\n".format(fname))
            exit()

    def save(self, fname='hcn'):
        """Store 
            - graph in <fname>.meta
            - parameters and state in <fname>.json
        """
        meta_fname = "{}-{}.meta".format(fname, self.step)
        json_fname = "{}-{}.json".format(fname, self.step)

        # save graph
        sys.stderr.write("<INFO> saving to '{}'\n".format(meta_fname))
        _saver = tf.train.Saver()
        _saver.save(self._sess, fname, global_step=self.step)
        
        # save state and options
        sys.stderr.write("<INFO> saving to '{}'\n".format(json_fname))
        with open(json_fname, 'w') as f:
            json.dump((self.opt, (self.state_c, self.state_h)), f)

    def shutdown(self):
        self._sess.close()

