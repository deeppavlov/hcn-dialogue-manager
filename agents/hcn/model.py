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

import sys
import copy
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class HybridCodeNetworkModel(object):

    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)

        # initialize parameters
        self.__init_params__()
# TODO: initialize action_size
        self.n_actions = params['action_size']
        # build computational graph
        self.__build__()
        # initialize metric variables
        self.reset_metrics()

        if self.opt.get('pretrained_model'):
            # restore state and session
            self.load()
        else:
            sys.stderr.write("<INFO> Initializing model from scratch.\n")
            # zero state
            self.reset_state()
            # initialize session
            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())


    def __init_params__(self, params=None):
        params = params or self.opt
        self.learning_rate = params['learning_rate']
        self.n_epoch = params['epoch_num']
        self.n_hidden = params['hidden_dim'] 

    def __build__(self):
        tf.reset_default_graph()

        # entry points
        self._features = tf.placeholder(tf.float32, [1, self.obs_size], 
                name='features')
        self._state_c = tf.placeholder(tf.float32, [1, self.n_hidden]) 
        self._state_h = tf.placeholder(tf.float32, [1, self.n_hidden]) 
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
        self._logits = tf.matmul(_state_reshaped, _Wo) + _bo

        # probabilities normalization : elemwise multiply with action mask
        self._probs = tf.multiply(tf.squeeze(tf.nn.softmax(self._logits)), 
                self._action_mask)
        
        self._prediction = tf.arg_max(self._probs, dimension=0)

        self._loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._logits, labels=self._action)

        self._train_op = tf.train.AdadeltaOptimizer(self.learning_rate)\
                .minimize(self._loss)

    def reset_metrics(self):
        self.n_examples = 0
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

    def train_step(self, features, action, action_mask):
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

    def forward(self, features, action_mask):
        probs, prediction, self.state_c, self.state_h = self._sess.run(
                [self._probs, self._prediction, self._state.c, self._state.h],
                feed_dict={
                    self._features: features.reshape([1, self.obs_size]),
                    self._state_c: self.state_c,
                    self._state_h: self.state_h,
                    seld._action_mask: action_mask
                    })
        return probs, prediction

    def load(self, fname=None):
# TODO: load hidden states
        fname = fname or self.opt['pretrained_model']
        fpath = 'ckpt/{}'.format(fname)
        _ckpt = tf.train.get_checkpoint_state('ckpt', fname)

        if _ckpt:
            sys.stderr.write("<INFO> loading model from {}\n".format(fpath))
            _saver = tf.train.Saver()
            _saver.restore(self._sess, fpath)
        else:
            sys.stderr.write("<ERR> checkpoint '{}' not found\n".format(fpath))

    def save(self, fname='hcn.ckpt'):
# TODO: save hidden states
        fpath = "ckpt/{}".format(fname) 
        
        sys.stderr.write("<INFO> saving to '{}'\n".format(fpath))
        _saver = tf.train.Saver()
        _saver.save(self._sess, fpath, global_step=0)

