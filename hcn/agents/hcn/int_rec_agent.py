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
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

from parlai.core.agents import Agent

import numpy as np
import sklearn.model_selection
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping

from sklearn.metrics import precision_recall_fscore_support, f1_score

import sys
sys.path.append('/home/dilyara/Documents/GitHub/general_scripts')
from random_search_class import param_gen
from save_load_model import init_from_scratch, init_from_saved, save
from fasttext_embeddings import text2embeddings
from metrics import fmeasure

from .embeddings_dict import EmbeddingsDict
from .int_rec_model import IntentRecognitionModel
import copy
import pickle
from . import config
import os, sys

class IntentRecognizerAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        """Add arguments from command line."""
        config.add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        """Initialize the class according to the given parameters in opt."""
        if opt['numthreads'] > 1:
            raise RuntimeError("numthreads > 1 not supported for this model.")

        self.id = self.__class__.__name__
        super().__init__(opt, shared)
        # to keep track of the episode
        self.episode_done = True

        # only create an empty dummy class when sharing
        if shared is not None:
            self.is_shared = True
            return

        # intialize parameters
        self.is_shared = False
        self.opt = copy.deepcopy(opt)

        self.intents = list(pickle.load(open(os.path.join(opt['datapath'],
                                                          'dstc2', 'intents.txt'), 'rb')))
        self.intents.append('unknown')
        self.n_classes = len(self.intents)
        self.confident_threshold = opt['intent_threshold']


        embedding_dict = EmbeddingsDict(self.opt, self.opt.get('embedding_dim'))

        self.model = IntentRecognitionModel(opt=self.opt, embedding_dict=embedding_dict)
        self.n_examples = 0
        print('___IntentRecognizerAgent and Model initialized___')



    def _predictions2text(self, predictions):
        # predictions: n_samples x n_classes
        y = [self.intents[np.where(sample > self.confident_threshold)[0]] for sample in predictions]
        return y

    def _text2predictions(self, predictions):
        eye = np.eye(self.n_classes)
        y = [np.sum([eye[class_id] for class_id in sample], axis=0) for sample in predictions]
        return y

    def observe(self, observation):
        """Receive an observation/action dict."""
        # observation = copy.deepcopy(observation)
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def _build_ex(self, ex):
        # check if empty input (end of epoch)
        if 'text' not in ex:
            return

        inputs = dict()
        inputs['question'] = ex['text']
        if 'labels' in ex:
            action = ex['act']
            slots = ex['slots']
            for slot in slots:
                if slot[0] == 'slot':
                    inputs['labels'] = action['act'] + '_' + slot[1]
                else:
                    inputs['labels'] = action['act'] + '_' + slot[0]

        return inputs

    def act(self):
        """Call batch act with batch of one sample."""
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):
        """Train model or predict for given batch of observations."""
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]
        predictions = [[] for _ in range(batch_size)]

        examples = [self.model._build_ex(obs) for obs in observations]
        valid_inds = [i for i in range(batch_size) if examples[i] is not None]
        examples = [ex for ex in examples if ex is not None]

        batch = self.model._batchify(examples)
        prediction = self.model.predict(batch)
        for i in range(len(prediction)):
            predictions[valid_inds[i]].append(prediction[i])

        for i in range(batch_size):
            prediction = predictions[i]
            batch_reply[i]['text'] = self._predictions2text([prediction])[0]
            batch_reply[i]['score'] = prediction

        return batch_reply

#------------------------------
    def fit_model(self, data, classes, to_use_kfold=False, verbose=True,
                  add_inputs=None, class_weight=None, shuffle=False):
        print("___Fitting model___")
        if class_weight is None:
            class_weight = [None for i in range(self.n_splits)]

        if to_use_kfold == True:
            print("___Stratified splitting data___")
            stratif_y = [np.nonzero(classes[j].values)[0][0] for j in range(data.shape[0])]
            kf_split = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
            kf_split.get_n_splits(data, stratif_y)
            for train_index, test_index in kf_split.split(data, stratif_y):
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = classes[train_index], classes[test_index]
                self.X_train.append(X_train)
                self.X_test.append(X_test)
                self.y_train.append(y_train)
                self.y_test.append(y_test)
        else:
            #this way data is a list of dataframes
            print("___Given %d splits of train data___" % self.n_splits)
            for i in range(self.n_splits):
                X_train = data[i]
                y_train = classes[i]
                self.X_train.append(X_train)
                self.y_train.append(y_train)


        if self.learning_parameters is None:
            print("___ERROR: learning parameters are not given___")
            exit(1)
        self.histories = []

        for model_ind in range(self.n_splits):
            if self.fasttext_embedding_model is not None:
                X_train_embed = text2embeddings(self.X_train[model_ind], self.fasttext_embedding_model, self.text_size, self.embedding_size)
            else:
                X_train_embed = self.X_train[model_ind]
            optimizer = Adam(lr=self.learning_parameters[model_ind]['lear_rate'],
                             decay=self.learning_parameters[model_ind]['lear_rate_decay'])
            self.models[model_ind].compile(loss='categorical_crossentropy',
                                           optimizer=optimizer,
                                           metrics=['categorical_accuracy',
                                           fmeasure])
            permut = np.random.permutation(np.arange(X_train_embed.shape[0]))
            if add_inputs is not None:
                self.histories.append(self.models[model_ind].fit([X_train_embed[permut], add_inputs[model_ind][permut]],
                                                                 self.y_train[model_ind][permut].reshape(-1, self.n_classes),
                                                                 batch_size=self.learning_parameters[model_ind]['batch_size'],
                                                                 epochs=self.learning_parameters[model_ind]['epochs'],
                                                                 validation_split=0.1,
                                                                 verbose=2 * verbose,
                                                                 shuffle=shuffle,
                                                                 class_weight=class_weight[model_ind],
                                                                 callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0)]))
            else:
                self.histories.append(self.models[model_ind].fit(X_train_embed[permut],
                                                                 self.y_train[model_ind][permut].reshape(-1,self.n_classes),
                                                                 batch_size=self.learning_parameters[model_ind]['batch_size'],
                                                                 epochs=self.learning_parameters[model_ind]['epochs'],
                                                                 validation_split=0.1,
                                                                 verbose=2 * verbose,
                                                                 shuffle=shuffle,
                                                                 class_weight=class_weight[model_ind],
                                                                 callbacks=[
                                                                     EarlyStopping(monitor='val_loss', min_delta=0.0)]))

        return True

    def predict(self, data=None, add_inputs=None):
        print("___Predictions___")
        if data is not None:
            X_test = data
        predictions = []

        if self.fasttext_embedding_model is not None:
            for model_ind in range(self.n_splits):
                X_test_embed = text2embeddings(X_test[model_ind], self.fasttext_embedding_model,
                                               self.text_size, self.embedding_size)
                if add_inputs is not None:
                    predictions.append(self.models[model_ind].predict([X_test_embed, add_inputs[model_ind]]).reshape(-1, self.n_classes))
                else:
                    predictions.append(self.models[model_ind].predict(X_test_embed).reshape(-1, self.n_classes))
            return predictions
        else:
            for model_ind in range(self.n_splits):
                X_test_embed = X_test[model_ind]
                if add_inputs is not None:
                    predictions.append(self.models[model_ind].predict([X_test_embed, add_inputs[model_ind]]).reshape(-1, self.n_classes))
                else:
                    predictions.append(self.models[model_ind].predict(X_test_embed).reshape(-1, self.n_classes))
            return predictions

    def report(self, true, predicts, mode=None):
        print("___Report___")
        if mode is not None:
            print("___MODE is %s___" % mode)

        f1_macro = f1_score(one_hot2ids(true), one_hot2ids(predicts), average='macro')
        f1_weighted = f1_score(one_hot2ids(true), one_hot2ids(predicts), average='weighted')
        print('F1 macro: %f', f1_macro)
        print('F1 weighted: %f', f1_weighted)
        print("%s \t %s \t%s \t %s \t %s" % ('type', 'precision', 'recall', 'f1-score', 'support'))
        f1_scores = []
        for ind, intent in enumerate(self.intents):
            scores = np.asarray(precision_recall_fscore_support(true[:, ind], np.round(predicts[:, ind])))[:, 1]
            print("%s \t %f \t %f \t %f \t %f" % (intent, scores[0], scores[1], scores[2], scores[3]))
            f1_scores.append(scores[2])
        return(f1_scores, f1_macro, f1_weighted)

    def all_params_to_dict(self):
        params_dict = dict()
        for model_ind in range(self.n_splits):
            for key in self.network_parameters[model_ind].keys():
                params_dict[key + '_' + str(model_ind)] = self.network_parameters[model_ind][key]
            for key in self.learning_parameters[model_ind].keys():
                params_dict[key + '_' + str(model_ind)] = self.learning_parameters[model_ind][key]
        return params_dict

    def save_models(self, fname):
        for model_ind in range(self.n_splits):
            save(self.models[model_ind],
                 fname=fname + '_' + str(model_ind))
        return True

    def get_tag_table(self, ner_data, tag_size):
        self.tag_size = tag_size
        list_of_tag_tables = []
        for model_ind in range(self.n_splits):
            tag_table = []
            for k in range(ner_data[model_ind].shape[0]):
                tags = [int(tag) for tag in ner_data[model_ind][k].split(' ')]
                request_tags = []
                for i_word, tag in enumerate(tags):
                    request_tags.append([(1 * (tag == m)) for m in range(self.tag_size)])
                tag_table.append(request_tags)
            list_of_tag_tables.append(tag_table)
        return list_of_tag_tables

    def one_hot2ids(self,one_hot_labels):
        return np.argmax(one_hot_labels, axis=1)















