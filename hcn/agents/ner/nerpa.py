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

import json

from parlai.core.agents import Agent

from . import config
from .model import NER
from fuzzywuzzy import process
from .corpus import Corpus


class NerProcessingAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        argparser.add_argument('--debug-wrong', type='bool', default=False,
                               help='Print debug output.')

        return argparser

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

        # intialize parameters
        self.is_shared = False

        # Load network parameters
        with open(opt['ner_params_filepath']) as f:
            network_params = json.load(f)

        # Create corpus object
        self._corpus = Corpus(dicts_filepath=opt['ner_dict_filepath'])

        # Build NER model
        self.model = NER(self._corpus,
                         pretrained_model_filepath=opt['ner_model_filepath'],
                         **network_params)

        # Load slots, vals, and vals variations
        with open(opt['ner_slot_vals_filepath']) as f:
            self._slot_vals = json.load(f)

    def observe(self, observation):
        """Receive an observation/action dict."""
        # observation = copy.deepcopy(observation)
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        """Update or predict on a single example (batchsize = 1)."""
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        if 'text' not in self.observation:
            raise RuntimeError('There is no "text" field in the observation!')

        # Predict slots and slot values
        if len(self.observation['text']) > 0:
            pred = self.predict_slots(self.observation['text'])
        else:
            pred = dict()
# TODO: add learning
        if 'labels' in self.observation:
            labels = self.observation['labels']
            if self.opt['debug_wrong'] and (labels != pred):
                print("True: '{}'\nPredicted: '{}'".format(
                    labels, pred))
        return pred

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = fname or self.opt.get('ner_model_file', None)
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

    def predict_slots(self, utterance):
        # For utterance extract named entities and perform normalization for slot filling

        # Naive tokenizer
        tokens = utterance.split()

        tags = self.model.predict_for_token_batch([tokens])[0]
        entities, slots = self._chunk_finder(tokens, tags)
        slot_values = dict()
        for entity, slot in zip(entities, slots):
            slot_values[slot] = self.ner2slot(entity, slot)
        return slot_values

    def ner2slot(self, input_entity, slot):
        # Given named entity return normalized slot value
        if isinstance(input_entity, list):
            input_entity = ' '.join(input_entity)
        entities = list()
        normalized_slot_vals = list()
        for entity_name in self._slot_vals[slot]:
            for entity in self._slot_vals[slot][entity_name]:
                entities.append(entity)
                normalized_slot_vals.append(entity_name)
        best_match = process.extract(input_entity, entities, limit=2 ** 20)[0][0]
        return normalized_slot_vals[entities.index(best_match)]

    @staticmethod
    def _chunk_finder(tokens, tags):
        # For BIO labeled sequence of tags extract all named entities form tokens
        # Example
        prev_tag = ''
        chunk_tokens = list()
        entities = list()
        slots = list()
        for token, tag in zip(tokens, tags):
            curent_tag = tag.split('-')[-1]
            current_prefix = tag.split('-')[0]
            if tag.startswith('B-'):
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = list()
                chunk_tokens.append(token)
            if current_prefix == 'I':
                if curent_tag != prev_tag:
                    if len(chunk_tokens) > 0:
                        entities.append(' '.join(chunk_tokens))
                        slots.append(prev_tag)
                        chunk_tokens = list()
                else:
                    chunk_tokens.append(token)
            if current_prefix == 'O':
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = list()
            prev_tag = curent_tag

        return entities, slots

