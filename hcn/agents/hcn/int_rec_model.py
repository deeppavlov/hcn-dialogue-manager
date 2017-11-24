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
from .int_rec_metrics import precision_recall_fscore_support, roc_auc_score

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)


class IntentRecognitionModel(object):

    def __init__(self, opt, embedding_dict=None):

        self.opt = copy.deepcopy(opt)
        self.kernel_sizes = [int(x) for x in opt['kernel_sizes_cnn'].split(' ')]
        self.from_saved = False
        np.random.seed(opt['model_seed'])
        tf.set_random_seed(opt['model_seed'])

        self.embedding_dict = embedding_dict if embedding_dict is not None \
            else EmbeddingsDict(opt, self.opt['embedding_dim'])

        self.intents = list(pickle.load(open(os.path.join(opt['datapath'],
                                                          'dstc2', 'intents.txt'), 'rb')))
        self.intents.append('unknown')
        self.intents = np.array(list(self.intents))
        self.n_classes = len(self.intents)
        self.confident_threshold = opt['intent_threshold']

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
        self.train_auc_m = 0.
        self.train_auc_w = 0.
        self.train_f1_m = 0.
        self.train_f1_w = 0.

        print("[ Considered intents:", self.intents, "]")
        print("[ Model initialized ]")

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
        # predictions: n_samples x n_classes, float values of probabilities
        # y = [self.intents[np.where(sample > self.confident_threshold)[0]]
        #      for sample in predictions]
        y = []
        for sample in predictions:
            to_add = np.where(sample > self.confident_threshold)[0]
            if len(to_add) > 0:
                y.append(self.intents[to_add])
            else:
                y.append([self.intents[np.argmax(sample)]])
        y = np.asarray(y)
        return y

    def _text2predictions(self, predictions):
        # predictions: list of lists with text intents in the reply
        eye = np.eye(self.n_classes)
        y = []
        for sample in predictions:
            curr = np.zeros(self.n_classes)
            # if (type(sample) is list or type(sample) is np.ndarray or type(sample) is tuple) \
            #         and len(sample) > 1:
            #     for intent in sample:
            #         curr += eye[np.where(self.intents == intent)[0]].reshape(-1)
            # else:
            #     curr = eye[np.where(self.intents == sample[0])[0]].reshape(-1)
            for intent in sample:
                curr += eye[np.where(self.intents == intent)[0]].reshape(-1)
            y.append(curr)
        y = np.asarray(y)
        return y

    def _batchify(self, batch):
        question = []
        for ex in batch:
            question.append(ex['question'])
        self.embedding_dict.add_items(question)
        embedding_batch = self.create_batch(question)

        if len(batch[0]) == 2:
            y = self._text2predictions([ex['labels'] for ex in batch])
            return embedding_batch, y
        else:
            return embedding_batch

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
        self.train_loss, self.train_acc = self.model.train_on_batch(x, y)
        y_pred = self.model.predict_on_batch(x).reshape(-1, self.n_classes)

        self.train_auc_m = roc_auc_score(y, y_pred, average='macro')
        self.train_auc_w = roc_auc_score(y, y_pred, average='weighted')

        y_pred_ = self._text2predictions(self._predictions2text(y_pred))
        self.train_f1_m = precision_recall_fscore_support(y, y_pred_, average='macro')
        self.train_f1_w = precision_recall_fscore_support(y, y_pred_, average='weighted')
        self.updates += 1
        return y_pred

    def predict(self, batch):
        y_pred = self.model.predict_on_batch(batch).reshape(-1, self.n_classes)
        print(y_pred[0])
        return y_pred

    def shutdown(self):
        self.embedding_dict = None

    def cnn_word_model(self):

        inp = Input(shape=(self.opt['max_sequence_length'], self.opt['embedding_dim'],))

        outputs = []
        for i in range(len(self.kernel_sizes)):
            output_i = Conv1D(self.opt['filters_cnn'], kernel_size=self.kernel_sizes[i],
                              activation=None, kernel_regularizer=l2(self.opt['regul_coef_conv']),
                              padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)

        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(self.opt['dense_dim'], activation=None,
                       kernel_regularizer=l2(self.opt['regul_coef_dense']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.opt['regul_coef_dense']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model
