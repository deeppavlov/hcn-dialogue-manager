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

import copy
import numpy as np

from parlai.core.agents import Agent

from . import config
from .model import HybridCodeNetworkModel
from .dict import ActionDictionaryAgent
from .entities import Babi5EntityTracker
from .utils import normalize_text, extract_babi5_template
from .metrics import DialogMetrics


class HybridCodeNetworkAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        HybridCodeNetworkAgent.dictionary_class().add_cmdline_args(argparser)
        argparser.add_argument('--debug', type='bool', default=False,
                help='Debug output.')

    @staticmethod
    def dictionary_class():
        return ActionDictionaryAgent

    def __init__(self, opt, shared=None):
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

        # initialize word dictionary and action templates
        self.word_dict = HybridCodeNetworkAgent.dictionary_class()(opt)

        # initialize entity tracker
        self.ent_tracker = Babi5EntityTracker()

        # intialize parameters
        self.is_shared = False
        self.n_actions = len(self.word_dict.action_templates)

        # initialize metrics
        self.metrics = DialogMetrics(self.n_actions)

        opt['action_size'] = self.n_actions
# TODO: train not only on bow and binary entity features
        opt['obs_size'] = len(self.word_dict) + self.ent_tracker.num_features 

        self.model = HybridCodeNetworkModel(opt) 

    def observe(self, observation):
        """Receive an observation/action dict."""
        observation = copy.deepcopy(observation)
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        """Update or predict on a single example (batchsize = 1)."""
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        reply = {'id': self.getID()}

        ex = self._build_ex(self.observation)
        if ex is None:
            return reply

        # either train or predict
        if 'labels' in self.observation:

            loss, pred = self.model.update(*ex)
            pred_text = self.word_dict.get_action_by_id(pred)

            # update metrics
            self.metrics.n_examples += 1
            if self.episode_done:
                self.metrics.n_dialogs += 1
            self.metrics.train_loss += loss
            self.metrics.conf_matrix[pred, ex[1]] += 1
            self.metrics.n_train_corr_examples += \
                    int(pred_text == self.observation['labels'][0])
#TODO: update number of correct dialogs
        else:
            if self.opt['debug']:
                print("Example = ", ex)
            probs, pred = self.model.predict(*ex)
            if self.opt['debug']:
                print("Probs = {}, pred = {}".format(probs, pred))
            reply['text'] = self.word_dict.get_action_by_id(pred)

        return reply

    def _build_ex(self, ex):
        # check if empty input (end of epoch)
        if not 'text' in ex:
            return

        # reinitilize entity tracker for new dialog
        if self.episode_done: 
            self.ent_tracker.restart()

        # tokenize input
        tokens = self.word_dict.tokenize(ex['text'])
 
        # Bag of words features
        bow_features = np.zeros(len(self.word_dict), dtype=np.float32)
        for t in tokens:
            bow_features[self.word_dict[t]] = 1.
        # Text entity features
        self.ent_tracker.update_entities(tokens)
        ent_features = self.ent_tracker.binary_features()
        if self.opt['debug']:
            print("Bow feats shape = {}, ent feats shape = {}".format(
                bow_features.shape, ent_features.shape))
        features = np.hstack((bow_features, ent_features))[np.newaxis, :]
        if self.opt['debug']:
            print("Feats shape = ", features.shape)
        
# TODO: non ones action mask
        action_mask = np.ones(self.n_actions, dtype=np.float32)
        if self.opt['debug']:
            print("Action_mask shape = ", action_mask.shape)
       
        # extract action templates
        targets = []
        for label in ex.get('labels', []):
            try:
                template = self.ent_tracker.extract_entity_types(
                        self.word_dict.tokenize(label))
                action = self.word_dict.get_action_id(template)
            except:
                raise RuntimeError('Invalid label. Should match one of action templates from train.')
            targets.append((label, action))
        # in case of prediction do not return action
        if self.opt['debug']:
            print("Targets = ", targets)
        if not targets:
            return (features, action_mask)

        # take only first label
        action = targets[0][1]

        return (features, action, action_mask)

    def report(self):
        return self.metrics.report()

    def reset_metrics(self):
        self.metrics.reset()

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = fname or self.opt.get('model_file', None)
        if fname:
            print("[saving model to {}]".format(fname))
            self.model.save(fname)
        else:
            print("[failed to save model]")

    def shutdown(self):
        """Final cleanup."""
        if not self.is_shared:
            if self.model is not None:
                self.model.shutdown()
            self.model = None

