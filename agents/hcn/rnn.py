#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class RNN():

    def __init__(self, obs_size, n_hidden=128, action_size=16):
        self.obs_size = obs_size
        self.n_hidden = n_hidden
        self.action_size = action_size

        # zero state
        self.reset_state()

        # build computational graph
        self.__build__()

        # initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

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
        Wi = tf.get_variable('Wi', [self.obs_size, self.n_hidden], 
                initializer=xavier_initializer())
        bi = tf.get_variable('bi', [self.n_hidden], 
                initializer=tf.constant_initializer(0.))

        # add relu/tanh here if necessary
        projected_features = tf.matmul(self._features, Wi) + bi

        lstm_f = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True)

        lstm_op, self.state = lstm_f(inputs=projected_features, 
                state=(self._state_c, self._state_h))

        # reshape LSTM's state tuple (2,128) -> (1,256)
        state_reshaped = tf.concat(axis=1, values=(self.state.c, self.state.h))

        # output projection
        Wo = tf.get_variable('Wo', [2*self.n_hidden, self.action_size], 
                initializer=xavier_initializer())
        bo = tf.get_variable('bo', [self.action_size], 
                initializer=tf.constant_initializer(0.))
        # get logits
        self.logits = tf.matmul(state_reshaped, Wo) + bo

        # probabilities normalization : elemwise multiply with action mask
        self.probs = tf.multiply(tf.squeeze(tf.nn.softmax(self.logits)), 
                self._action_mask)
        
        self.prediction = tf.arg_max(self.probs, dimension=0)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self._action)

        self.train_op = tf.train.AdadeltaOptimizer(0.1).minimize(self.loss)

    def reset_state(self):
        # set zero state
        self.state_c = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.state_h = np.zeros([1,self.nb_hidden], dtype=np.float32)

    def train_step(self, features, action, action_mask):
        _, loss_value, self.state_c, self.state_h = self.sess.run(
                [self.train_op, self.loss, self.state.c, self.state.h],
                feed_dict={
                    self._features: features.reshape([1, self.obs_size]),
                    self._action: [action],
                    self._state_c: self.state_c,
                    self._state_h: self.state_h,
                    self._action_mask: action_mask
                    })
        return loss_value

    def forward(self, features, action_mask):
        probs, prediction, self.state_c, self.state_h = self.sess.run(
                [self.probs, self.prediction, self.state.c, self.state.h],
                feed_dict={
                    self._features: features.reshape([1, self.obs_size]),
                    self._state_c: self.state_c,
                    self._state_h: self.state_h,
                    seld._action_mask: action_mask
                    })
        return probs, prediction

    def load(self, fname='hcn.ckpt'):
        fpath = 'ckpt/{}'.format(fname)
        ckpt = tf.train.get_checkpoint_state('ckpt', fname)

        if ckpt:
            sys.stderr.write("load from {}\n".format(fpath))
            saver = tf.train.Saver()
            saver.restore(self.sess, fpath)
        else:
            sys.stderr.write("<ERR> checkpoint '{}' not found.\n".format(fpath))

    def save(self, fname='hcn.ckpt'):
        saver = tf.train.Saver()
        saver.save(self.sess, fpath, global_step=0)
        sys.stderr.write("save to {}\n".format(fpath))

