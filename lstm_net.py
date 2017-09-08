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

        # construct computational graph
        self.__build__()

        # initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # zero init states
        self.reset_state()

    def __build__(self):
        tf.reset_default_graph()

        # entry points
        self._features = tf.placeholder(tf.float32, [1, self.obs_size], name='features')
        self._init_state_c, self._init_state_h = ( 
                tf.placeholder(tf.float32, [1, self.n_hidden]) 
                for _ in range(2) )
        self._action = tf.placeholder(tf.int32, name='ground_truth_action')
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
                state=(self._init_state_c, self._init_state_h))

        # reshape LSTM's state tuple (2,128) -> (1,256)
        state_reshaped = tf.concat(axis=1, values=(self.state.c, self.state.h))

        # output projection
        Wo = tf.get_variable('Wo', [2*self.n_hidden, self.action_size], 
                initializer=xavier_initializer())
        bo = tf.get_variable('bo', [self.action_size], 
                initializer=tf.constant_initializer(0.))
        # get logits
        self.logits = tf.matmul(state_reshaped, Wo) + bo
        # probabilities
        #  normalization : elemwise multiply with action mask
        self.probs = tf.multiply(tf.squeeze(tf.nn.softmax(self.logits)), 
                self._action_mask)
        
        # prediction
        self.prediction = tf.arg_max(self.probs, dimension=0)

        # loss
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, 
                labels=self._action)

        # train op
        self.train_op = tf.train.AdadeltaOptimizer(0.1).minimize(self.loss)

    def reset_state(self):
        # set init state to zeros
        self.init_state_c = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1,self.nb_hidden], dtype=np.float32)

