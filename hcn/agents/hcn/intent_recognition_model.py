# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

from .metrics import roc_auc_score
import os
import numpy as np
import copy
from keras.layers import Dense, Activation, Input, concatenate
from keras.models import Model

from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from sklearn import linear_model, svm
from keras import backend as K
from keras.metrics import binary_accuracy
import json
import pickle

from .embeddings_dict import EmbeddingsDict

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)


class IntentRecognitionModel(object):

    def __init__(self, opt):

        self.embedding_dict = None
        self.opt = copy.deepcopy(opt)
        self.kernel_sizes = [int(x) for x in opt['kernel_sizes_cnn'].split(' ')]
        self.from_saved = False
        np.random.seed(opt['model_seed'])
        tf.set_random_seed(opt['model_seed'])

        self.embedding_dict = EmbeddingsDict(opt, self.opt['embedding_dim'])

        if self.opt.get('model_file') and (os.path.isfile(opt['model_file'] + '.h5')) and \
                (os.path.isfile(opt['model_file'] + '_opt.json')):
            print('[Initializing model from saved]')
            self.from_saved = True
            self._init_from_saved(opt['model_file'])
        else:
            if self.opt.get('pretrained_model'):
                print('[Initializing model from pretrained]')
                self.from_saved = True
                self._init_from_saved(opt['pretrained_model'])
            else:
                print('[ Initializing model from scratch ]')
                self._init_from_scratch()

        self.n_examples = 0
        self.updates = 0
        self.train_loss = 0.0
        self.train_acc = 0.0
        self.train_auc = 0.0
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.val_auc = 0.0

        self.intents = None
        self.n_classes = None
        self.confident_threshold = opt['confident_threshold']

        with open(os.path.join(opt['datapath'], 'dstc2', "intents.txt"), "rb") as fp:
            self.intents = pickle.load(fp)
            self.n_classes = len(self.intents)

    def _init_from_scratch(self):
        self.model = self.cnn_word_model()
        optimizer = Adam(lr=self.opt['learning_rate'], decay=self.opt['learning_decay'])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['categorical_accuracy'])

    def save(self, fname=None):
        fname = self.opt.get('model_file', None) if fname is None else fname

        if fname:
            print("[ saving model: " + fname + " ]")
            self.model.save_weights(fname + '.h5')
            self.embedding_dict.save_items(fname)

            with open(fname + '_opt.json', 'w') as opt_file:
                json.dump(self.opt, opt_file)

    def _init_from_saved(self, fname):

        with open(fname + '_opt.json', 'r') as opt_file:
            self.opt = json.load(opt_file)

        self.model = self.cnn_word_model()
        optimizer = Adam(lr=self.opt['learning_rate'], decay=self.opt['learning_decay'])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['categorical_accuracy'])
            print('[ Loading model weights %s ]' % fname)
            self.model.load_weights(fname + '.h5')

    def _build_ex(self, ex):

        if 'text' not in ex:
            return
        inputs = dict()
        inputs['question'] = ex['text']
        if 'labels' in ex:
            inputs['labels'] = ex['labels']

        return inputs

    def _predictions2text(self, predictions):
        # predictions: n_samples x n_classes
        y = [self.intents[np.where(sample > self.confident_threshold)[0]] for sample in predictions]
        return y

    def _text2predictions(self, predictions):
        eye = np.eye(self.n_classes)
        y = [np.sum([eye[class_id] for class_id in sample], axis=0) for sample in predictions]
        return y

    def _batchify(self, batch, word_dict=None):
        if self.model_type == 'nn':
            question = []
            for ex in batch:
                question.append(ex['question'])
            self.embedding_dict.add_items(question)
            embedding_batch = self.create_batch(question)

            if len(batch[0]) == 2:
                y = [1 if ex['labels'][0] == 'Insult' else 0 for ex in batch]
                return embedding_batch, y
            else:
                return embedding_batch

        if self.model_type == 'ngrams':
            question = []
            for ex in batch:
                question.append(ex['question'])

            if len(batch[0]) == 2:
                y = [1 if ex['labels'][0] == 'Insult' else 0 for ex in batch]
                return question, y
            else:
                return question

    def create_batch(self, sentence_li):
        embeddings_batch = []
        for sen in sentence_li:
            embeddings = []
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            if len(tokens) > self.opt['max_sequence_length']:
                tokens = tokens[:self.opt['max_sequence_length']]
            for tok in tokens:
                embeddings.append(self.embedding_dict.tok2emb.get(tok))
            if len(tokens) < self.opt['max_sequence_length']:
                pads = [np.zeros(self.opt['embedding_dim'])
                        for _ in range(self.opt['max_sequence_length'] - len(tokens))]
                embeddings = pads + embeddings
            embeddings = np.asarray(embeddings)
            embeddings_batch.append(embeddings)
        embeddings_batch = np.asarray(embeddings_batch)
        return embeddings_batch

    def update(self, batch):
        x, y = batch
        y = np.array(y)
        y_pred = None

        if self.model_type == 'nn':
            self.train_loss, self.train_acc = self.model.train_on_batch(x, y)
            y_pred = self.model.predict_on_batch(x).reshape(-1)
            self.train_auc = roc_auc_score(y, y_pred)

        if self.model_type == 'ngrams':
            x = vectorize_select_from_data(x, self.vectorizers, self.selectors)
            self.model.fit(x, y.reshape(-1))
            y_pred = np.array(self.model.predict_proba(x)[:,1]).reshape(-1)
            y_pred_tensor = K.constant(y_pred, dtype='float64')
            self.train_loss = K.eval(binary_crossentropy(y.astype('float'), y_pred_tensor))
            self.train_acc = K.eval(binary_accuracy(y.astype('float'), y_pred_tensor))
            self.train_auc = roc_auc_score(y, y_pred)
        self.updates += 1
        return y_pred

    def predict(self, batch):
        if self.model_type == 'nn':
            y_pred = np.array(self.model.predict_on_batch(batch)).reshape(-1)
            return y_pred
        if self.model_type == 'ngrams':
            x = vectorize_select_from_data(batch, self.vectorizers, self.selectors)
            predictions = np.array(self.model.predict_proba(x)[:,1]).reshape(-1)
            return predictions

    def shutdown(self):
        self.embedding_dict = None

    def cnn_word_model(self):

        inp = Input(shape=(self.opt['max_sequence_length'], self.opt['embedding_dim'],))

        outputs = []
        for i in range(len(self.kernel_sizes)):
            output_i = Conv1D(self.opt['filters_cnn'], kernel_size=self.kernel_sizes[i], activation=None,
                              kernel_regularizer=l2(self.opt['regul_coef_conv']), padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)
        print('Concatenate shape:', output.shape)

        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(self.opt['dense_dim'], activation=None,
                       kernel_regularizer=l2(self.opt['regul_coef_dense']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None, kernel_regularizer=l2(self.opt['regul_coef_dense']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

