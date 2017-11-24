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
        self.intents = np.array(list(self.intents))
        self.n_classes = len(self.intents)
        self.confident_threshold = opt['intent_threshold']

        embedding_dict = EmbeddingsDict(self.opt, self.opt.get('embedding_dim'))

        self.model = IntentRecognitionModel(opt=self.opt, embedding_dict=embedding_dict)
        self.n_examples = 0
        print('[ IntentRecognizerAgent and Model initialized ]')

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

    def observe(self, observation):
        """Gather obtained observation (sample) with previous observations."""
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

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

        if 'labels' in observations[0]:
            self.n_examples += len(examples)
            batch = self.model._batchify(examples)
            predictions = self.model.update(batch)
            predictions_text = self._predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions_text[i]
                batch_reply[valid_inds[i]]['score'] = predictions[i]
            print(examples[0])
            print(batch[0][0])
            print(predictions[0])
            print(predictions_text[0])
        else:
            batch = self.model._batchify(examples)
            predictions = self.model.predict(batch)
            predictions_text = self._predictions2text(predictions)

            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions_text[i]
                batch_reply[valid_inds[i]]['score'] = predictions[i]

            print(examples[0])
            print(batch[0])
            print(predictions[0])
            print(predictions_text[0])
        return batch_reply

    def report(self):
        """Return report."""
        report = dict()
        report['updates'] = self.model.updates
        report['n_examples'] = self.n_examples
        report['loss'] = self.model.train_loss
        report['categ_accuracy'] = self.model.train_acc
        report['auc_macro'] = self.model.train_auc_m
        report['auc_weighted'] = self.model.train_auc_w
        report['f1_macro'] = self.model.train_f1_m
        report['f1_weighted'] = self.model.train_f1_w
        return report

    def save(self):
        """Save trained model."""
        self.model.save()
