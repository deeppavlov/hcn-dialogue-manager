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

import os
import copy
import numpy as np
import re

from parlai.core.agents import Agent, create_agent
from parlai.core.params import ParlaiParser

from . import config
from . import tracker
from . import templates as tmpl
from .emb_dict import EmbeddingsDict
from .model import HybridCodeNetworkModel
from .preprocess import HCNPreprocessAgent
from .metrics import DialogMetrics


class HybridCodeNetworkAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        HybridCodeNetworkAgent.dictionary_class().add_cmdline_args(argparser)
        return argparser

    @staticmethod
    def dictionary_class():
        return HCNPreprocessAgent

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

        # load templates if present
        self.templates = None
        if self.opt['template_path_relative']:
            self.opt['template_file'] = os.path.join(
                self.opt['datapath'],
                *os.path.split(self.opt['template_file'])
            )
        self.templates = tmpl.Templates().load(self.opt['template_file'])
        print("[using {} provided templates from `{}`]"\
                .format(len(self.templates), self.opt['template_file']))

        # initialize word dictionary, action templates and embeddings
        self.preps = HybridCodeNetworkAgent.dictionary_class()(opt)

        # initialize embeddings
        self.embeddings = None
        if self.opt.get("embedding_file") is not None:
            print("[loading embeddings from `{}`]"\
                  .format(self.opt['embedding_file']))
            self.embeddings = EmbeddingsDict(opt)
        else:
            print("[no embeddings provided]")

        # initialize tracker
        self.tracker = tracker.DefaultTracker(self.preps.slot_names)

        # initialize slot filler
        self.slot_model = self._load_slot_model()

        # intialize parameters
        self.is_shared = False
        self.db_result = None
        self.n_actions = len(self.templates)
        self.emb_size = 0
        if self.embeddings is not None:
            self.emb_size = self.embeddings.dim
        self.prev_action = np.zeros(self.n_actions, dtype=np.float32)

        # initialize metrics
        self.metrics = DialogMetrics(self.n_actions)

        opt['action_size'] = self.n_actions
        opt['obs_size'] = 4 + len(self.preps.words) + self.emb_size +\
                2 * self.tracker.state_size + self.n_actions

        self.model = HybridCodeNetworkModel(opt)

    def _load_slot_model(self):
        if self.opt.get('slot_model') is not None:
            print("[loading `{}` slot classifier]".format(self.opt['slot_model']))
            opts = ['-m', self.opt['slot_model']]
            parser = ParlaiParser(True)
            parser.add_model_args(opts)
            return create_agent(parser.parse_args(opts))
        print("[no slot classifier provided]")
        return None

    def _get_slots(self, text):
        if self.slot_model is not None:
            self.slot_model.observe({
                'text': text,
                'episode_done': True
            })
            return self.slot_model.act()
        return {}

    def observe(self, observation):
        """Receive an observation/action dict."""
        # observation = copy.deepcopy(observation)
        self.observation = observation
        self.episode_done = observation['episode_done']
        if 'db_result' in observation:
            self.db_result = observation['db_result']
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
            self.prev_action *= 0.
            self.prev_action[pred] = 1.

            pred_text = self._generate_response(pred).lower()
            label_text = self.observation['labels'][0].lower()
            label_text = self.preps.words.detokenize(label_text.split())

            # update metrics
            self.metrics.n_examples += 1
            if self.episode_done:
                self.metrics.n_dialogs += 1
            self.metrics.train_loss += loss
            self.metrics.conf_matrix[pred, ex[1]] += 1
            self.metrics.n_train_corr_examples += int(pred_text == label_text)
# TODO: update number of correct dialogs
        else:
            probs, pred = self.model.predict(*ex)
            self.prev_action *= 0.
            self.prev_action[pred] = 1.
            reply['text'] = self._generate_response(pred)

        # reinitilize entity tracker for new dialog
        if self.episode_done:
            self.tracker.reset_state()
            self.db_result = None
            self.prev_action *= 0.
            self.model.reset_state()

        return reply

    def _build_ex(self, ex):
        # check if empty input (end of epoch)
        if 'text' not in ex:
            return

        # tokenize input
        tokens = self.preps.words.tokenize(ex['text'])

        # Bag of words features
        bow_features = np.zeros(len(self.preps.words), dtype=np.float32)
        for t in tokens:
            bow_features[self.preps.words[t]] = 1.

        # Embeddings
        emb_features = np.array([], dtype=np.float32)
        if self.embeddings is not None:
            emb_features = self.embeddings.encode(tokens)

        # Text entity features
        prev_slots = self.tracker.get_slots()
        self.tracker.update_slots(self._get_slots(' '.join(tokens)))

        binary_features = self.tracker.binary_features()
        diff_features = self.tracker.diff_features(prev_slots)
        ent_features = np.hstack((binary_features, diff_features))
        # Other features
        context_features = np.array([
            sum(binary_features),
            sum(diff_features),
            (self.observation.get('db_result') == {}) * 1.,
            (self.db_result == {}) * 1.
            ], dtype=np.float32)
        features = np.hstack((
            bow_features, emb_features, ent_features, context_features,
            self.prev_action
        ))[np.newaxis, :]

        # constructing mask of allowed actions
        action_mask = np.ones(self.n_actions, dtype=np.float32)
        if self.opt['action_mask']:
# TODO: non-ones action mask
            for a_id in range(self.n_actions):
                tmpl = str(self.templates.templates[a_id])
                for entity in re.findall('#{}', tmpl):
                    if entity not in self.tracker.get_slots()\
                       and entity not in (self.db_result or {}):
                        action_mask[a_id] = 0

        # extract action templates
        targets = []
        if self.templates and ex.get('act') is not None:
            label = ex.get('labels', [''])[0]
            targets.append((label, self.templates.actions.index(ex['act'])))

        # in case of prediction do not return action
        if not targets:
            return (features, action_mask)

        # take only first label
        action = targets[0][1]

        return (features, action, action_mask)

    def _generate_response(self, action_id):
        """
        Convert action template id and entities from tracker
        to final response.
        """
        template = self.templates.templates[int(action_id)]

        state = self.tracker.get_slots()
        if self.db_result is not None:
            for k, v in self.db_result.items():
                state[k] = str(v)

        return template.generate_text(state)

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
