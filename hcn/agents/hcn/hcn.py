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
from .entities import Babi5EntityTracker, Babi6EntityTracker
from .utils import normalize_text
from .utils import is_silence, is_null_api_answer, is_api_answer
from .metrics import DialogMetrics


class HybridCodeNetworkAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        HybridCodeNetworkAgent.dictionary_class().add_cmdline_args(argparser)
        argparser.add_argument('--debug', type='bool', default=False,
                help='Print debug output.')
        argparser.add_argument('--debug-wrong', type='bool', default=False,
                help='Print debug output.')
        argparser.add_argument('--tracker', required=True, 
                choices=['babi5', 'babi6'],
                help='Type of entity tracker to use. Implemented only for dialog_babi5 and dialog_babi6.')

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
        self.ent_tracker = None
        if self.opt['tracker'] == 'babi5':
            self.ent_tracker = Babi5EntityTracker()
        elif self.opt['tracker'] == 'babi6':
            self.ent_tracker = Babi6EntityTracker()

        # intialize parameters
        self.is_shared = False
        self.database_results = []
        self.current_result = None
        self.n_actions = len(self.word_dict.action_templates)

        # initialize metrics
        self.metrics = DialogMetrics(self.n_actions)

        opt['action_size'] = self.n_actions
# TODO: train not only on bow and binary entity features
        opt['obs_size'] = 3 + len(self.word_dict) + self.ent_tracker.num_features 

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
            pred_text = self._generate_response(pred)
            label_text = self.observation['labels'][0]
            label_text = self.word_dict.detokenize(label_text.split())

            # update metrics
            self.metrics.n_examples += 1
            if self.episode_done:
                self.metrics.n_dialogs += 1
            self.metrics.train_loss += loss
            self.metrics.conf_matrix[pred, ex[1]] += 1
            self.metrics.n_train_corr_examples += int(pred_text == label_text)
            if self.opt['debug_wrong'] and (pred_text != label_text) and pred != 6:
                print("True: '{}'\nPredicted: '{}'".format(label_text, pred_text))
#TODO: update number of correct dialogs
        else:
            if self.opt['debug']:
                print("Example = ", ex)
            probs, pred = self.model.predict(*ex)
            if self.opt['debug']:
                print("Probs = {}, pred = {}".format(probs, pred))
                print("Entities = ", self.ent_tracker.tracked.values())
            reply['text'] = self._generate_response(pred)

        return reply

    def _build_ex(self, ex):
        # check if empty input (end of epoch)
        if not 'text' in ex:
            return

        # reinitilize entity tracker for new dialog
        if self.episode_done: 
            self.ent_tracker.restart()
            self.database_results = []
            self.current_result = None

        # tokenize input
        tokens = self.word_dict.tokenize(ex['text'])
 
        # store database results
        if is_api_answer(ex['text']):
            self.word_dict.update_database(ex['text'])
            if self.opt['debug']:
                print("Parsed api result = ", self.database_results)

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
        # Other features
        context_features = np.array(
                [is_silence(ex['text']),
                    is_api_answer(ex['text']),
                    is_null_api_answer(ex['text'])], 
                dtype=np.float32)
        features = np.hstack((bow_features, ent_features, context_features))\
                [np.newaxis, :]
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

    def _generate_response(self, action_id):
        """Convert action template id and entities from tracker to final response."""
        # is api request
        if action_id == 1:
            self.database_results = self.word_dict.database.search(
                    self.ent_tracker.tracked,
                    order_by='R_rating', ascending=False)
            if self.opt['debug']:
                print("DatabaseSimulator results = ", self.database_results)
        # is restaurant offering
        if (action_id == 12) and self.database_results:
            self.current_result = self.database_results.pop(0)
            if self.opt['debug']:
                print("API best response = ", self.current_result)

        template = self.word_dict.get_action_by_id(action_id)
        if self.current_result is not None:
            for k, v in self.current_result.items():
                template = template.replace(k, str(v)) 
        return self.ent_tracker.fill_entities(template) 

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

