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

import numpy as np
import sklearn.model_selection
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from metrics import fmeasure
from sklearn.metrics import precision_recall_fscore_support, f1_score

import sys
sys.path.append('/home/dilyara/Documents/GitHub/general_scripts')
from random_search_class import param_gen
from save_load_model import init_from_scratch, init_from_saved, save
from fasttext_embeddings import text2embeddings


def one_hot2ids(one_hot_labels):
    return np.argmax(one_hot_labels, axis=1)


class IntentRecognizer(object):

    # data - list or array of strings-request len = N_samples
    # classes - np.array of one-hot classes N_samples x n_classes

    # IF to_use_kfold = False:
    # data - list of lists or arrays of strings-request len = N_samples
    # classes - list of arrays of one-hot classes N_samples x n_classes

    def __init__(self, intents, n_splits=None, fasttext_embedding_model=None):

        self.intents = intents
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.network_parameters = None
        self.learning_parameters = None
        self.n_classes = len(intents)
        self.n_splits = n_splits
        self.tag_size = None
        self.text_size = None
        self.embedding_size = None
        self.kernel_sizes = None
        self.models = None
        self.model_function = None
        self.histories = None
        self.fasttext_embedding_model = None

        if fasttext_embedding_model is not None:
            print("___Fasttext embedding model is loaded___")
            self.fasttext_embedding_model = fasttext_embedding_model
        print('___Recognizer initialized___')

    def gener_network_parameters(self, **kwargs):
        print("___Considered network parameters___")
        self.network_parameters = []
        for i in range(self.n_splits):
            self.network_parameters.append(param_gen(**kwargs))    #generated dict
            print(self.network_parameters[-1])
        return True

    def gener_learning_parameters(self, **kwargs):
        print("___Considered learning parameters___")
        self.learning_parameters = []
        for i in range(self.n_splits):
            self.learning_parameters.append(param_gen(**kwargs))   #generated dict
            print(self.learning_parameters[-1])
        return True

    def init_network_parameters(self, arg_list):
        print("___Considered network parameters___")
        self.network_parameters = arg_list                 #dict
        print(self.network_parameters)
        return True

    def init_learning_parameters(self, arg_list):
        print("___Considered learning parameters___")
        self.learning_parameters = arg_list                #dict
        print(self.learning_parameters)
        return True

    def init_model(self, model_function, text_size, embedding_size, kernel_sizes, add_network_params=None):
        self.model_function = model_function
        print("___Model initialized____")
        if self.network_parameters is None:
            print("___ERROR: network parameters are not given___")
            exit(1)

        self.text_size = text_size
        self.embedding_size = embedding_size
        self.kernel_sizes = kernel_sizes

        self.models = []
        for model_ind in range(self.n_splits):
            if add_network_params is not None:
                self.models.append(init_from_scratch(self.model_function, text_size=self.text_size, n_classes=self.n_classes,
                                                     embedding_size=self.embedding_size,
                                                     kernel_sizes=self.kernel_sizes,
                                                     **add_network_params,
                                                     **(self.network_parameters[model_ind])))
            else:
                self.models.append(init_from_scratch(self.model_function, text_size=self.text_size, n_classes=self.n_classes,
                                                     embedding_size=self.embedding_size,
                                                     kernel_sizes=self.kernel_sizes,
                                                     **(self.network_parameters[model_ind])))
        return True

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















