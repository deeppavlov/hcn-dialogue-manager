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
import copy
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import LSTMStateTuple


class HybridCodeNetworkModel(object):

    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)

        if self.opt.get('pretrained_model'):
            print("[ Initializing model from `{}` ]".format(self.opt['pretrained_model']))
            # restore state, parameters and session
            self.restore(self.opt['pretrained_model'])
        else:
            print("[ Initializing model from scratch ]")
            # initialize parameters
            self.__init_params__()
            # build computational graph
            self.__build__()
            # initialize session
            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())
            # zero state
            self.reset_state()

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
        self._action_mask = tf.placeholder(tf.float32, [self.n_actions], 
                name='action_mask')

        # input projection
        _Wi = tf.get_variable('Wi', [self.obs_size, self.n_hidden], 
                initializer=xavier_initializer())
        _bi = tf.get_variable('bi', [self.n_hidden], 
                initializer=tf.constant_initializer(0.))

        # add relu/tanh here if necessary
        _projected_features = tf.matmul(self._features, _Wi) + _bi

        _lstm_f = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True)

        _lstm_op, self._next_state = _lstm_f(inputs=_projected_features, 
                state=(self._state_c, self._state_h))

        # reshape LSTM's state tuple (2,128) -> (1,256)
        _state_reshaped = tf.concat(axis=1, 
                values=(self._next_state.c, self._next_state.h))

        # output projection
        _Wo = tf.get_variable('Wo', [2*self.n_hidden, self.n_actions], 
                initializer=xavier_initializer())
        _bo = tf.get_variable('bo', [self.n_actions], 
                initializer=tf.constant_initializer(0.))
        # get logits
        _logits = tf.matmul(_state_reshaped, _Wo) + _bo

        # probabilities normalization : elemwise multiply with action mask
        self._probs = tf.multiply(tf.squeeze(tf.nn.softmax(_logits)), 
                self._action_mask)
        
        self._prediction = tf.argmax(self._probs, axis=0)

        self._loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=_logits, labels=self._action)

        self._step = tf.Variable(0, trainable=False, name='global_step') 
        self._train_op = tf.train.AdadeltaOptimizer(self.learning_rate)\
                .minimize(self._loss, global_step=self._step)

        # create collections for easier `restore` operation
        tf.add_to_collection('next_state_c', self._next_state.c)
        tf.add_to_collection('next_state_h', self._next_state.h)
        tf.add_to_collection('probs', self._probs)
        tf.add_to_collection('prediction', self._prediction)
        tf.add_to_collection('loss', self._loss)
        tf.add_to_collection('global_step', self._step)
        tf.add_to_collection('train_op', self._train_op)

    def reset_state(self):
        # set zero state
        self.state_c = np.zeros([1,self.n_hidden], dtype=np.float32)
        self.state_h = np.zeros([1,self.n_hidden], dtype=np.float32)

    def update(self, features, action, action_mask):
        _, loss_value, self.state_c, self.state_h, prediction = \
                self._sess.run(
                        [ self._train_op, self._loss, self._next_state.c, 
                            self._next_state.h, self._prediction ],
                        feed_dict={
                            self._features: features.reshape([1, self.obs_size]),
                            self._action: [action],
                            self._state_c: self.state_c,
                            self._state_h: self.state_h,
                            self._action_mask: action_mask
                            })
        return loss_value[0], prediction

    def predict(self, features, action_mask):
        probs, prediction, self.state_c, self.state_h = \
                self._sess.run(
                        [ self._probs, self._prediction, self._next_state.c, 
                            self._next_state.h ],
                        feed_dict={
                            self._features: features.reshape([1, self.obs_size]),
                            self._state_c: self.state_c,
                            self._state_h: self.state_h,
                            self._action_mask: action_mask
                            })
        return probs, prediction

    def restore(self, fname=None):
        """Restore graph, important operations and state"""
        fname = fname or self.opt['pretrained_model']
        json_fname, meta_fname = "{}.json".format(fname), "{}.meta".format(fname)
        #tf.train.get_checkpoint_state(*os.path.split(meta_fname))

        if os.path.isfile(meta_fname):
            print("[restoring model from {}]".format(meta_fname))
            tf.reset_default_graph()
            #_saver = tf.train.Saver()
            _saver = tf.train.import_meta_graph(meta_fname)
            self._sess = tf.Session()
            _saver.restore(self._sess, fname)

            # restore placeholders
            _graph = tf.get_default_graph()
            self._features = _graph.get_tensor_by_name("features:0")
            self._state_c = _graph.get_tensor_by_name("state_c:0")
            self._state_h = _graph.get_tensor_by_name("state_h:0")
            self._action = _graph.get_tensor_by_name("ground_truth_action:0")
            self._action_mask = _graph.get_tensor_by_name("action_mask:0")

            # restore important operations
            self._next_state = LSTMStateTuple(
                    _graph.get_collection('next_state_c')[0],
                    _graph.get_collection('next_state_h')[0]
                    )
            self._probs = _graph.get_collection('probs')[0]
            self._prediction = _graph.get_collection('prediction')[0]
            self._loss = _graph.get_collection('loss')[0]
            self._train_op = _graph.get_collection('train_op')[0]
            self._step = _graph.get_collection('global_step')[0]

            # restore state
            if os.path.isfile(json_fname):
                with open(json_fname, 'r') as f:
                    params, (state_c, state_h) = json.load(f)
                    self.state_c = np.array(state_c, dtype=np.float32)
                    self.state_h = np.array(state_h, dtype=np.float32)
                    self.__init_params__(params)
            else:
                print("[{} not found]".format(json_fname))
                exit()
        else:
            print("[{} not found]".format(meta_fname))
            exit()

    def save(self, fname='hcn'):
        """Store 
            - graph in <fname>.meta
            - parameters and state in <fname>.json
        """
        step = tf.train.global_step(self._sess, self._step)
        meta_fname = "{}-{}.meta".format(fname, step)
        json_fname = "{}-{}.json".format(fname, step)

        # save graph
        print("[saving graph to {}]".format(meta_fname))
        _saver = tf.train.Saver()
        _saver.save(self._sess, fname, global_step=step)
        
        # save state and options
        print("[saving options to {}]".format(json_fname))
        with open(json_fname, 'w') as f:
            json.dump((self.opt, 
                (self.state_c.tolist(), self.state_h.tolist())), f)

    def shutdown(self):
        self._sess.close()

