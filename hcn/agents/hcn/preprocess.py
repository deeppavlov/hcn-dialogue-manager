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
import json

from parlai.core.dict import DictionaryAgent, Agent

from .utils import normalize_text
from .utils import is_api_answer, is_null_api_answer, iter_api_response
from .dict import WordDictionaryAgent, ActionDictionaryAgent


class HCNPreprocessAgent(Agent):
    """Contains WordDictionaryAgent and ActionDictionaryAgent."""

    @staticmethod
    def add_cmdline_args(argparser):
        WordDictionaryAgent.add_cmdline_args(argparser)
        ActionDictionaryAgent.add_cmdline_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        self.id = self.__class__.__name__

        # intialize action dictionary
        self.actions = ActionDictionaryAgent(opt, shared)

        # word dictionary
        self.words = WordDictionaryAgent(opt, shared)

        # track all slot names
        self.slot_names = []
        if opt.get('dict_file') is not None\
           and os.path.isfile(opt['dict_file'] + '.slots'):
            self.slot_names = json.load(open(opt['dict_file'] + '.slots', 'r'))
        elif opt.get('pretrained_model') is not None\
            and os.path.isfile(opt['model_file'] + '.dict.slots'):
            self.slot_names = json.load(open(opt['pretrained_model'] + '.dict.slots', 'r'))
        elif opt.get('model_file') is not None\
            and os.path.isfile(opt['model_file'] + '.dict.slots'):
            self.slot_names = json.load(open(opt['model_file'] + '.dict.slots', 'r'))


    def act(self):
#TODO: update documentation
        """
            - Add words passed in the 'text' field of the observation to
        the dictionary,
            - extract action templates from all 'label_candidates' once
        """
        # add to word dict
        self.words.observe(self.observation)
        self.words.act()

        # add to action dict
        self.actions.observe(self.observation)
        self.actions.act()

        # is `intents` in observation, save slot names
        for intent in self.observation.get('intents', []):
            for slot, value in intent.get('slots', []):
                if slot not in self.slot_names:
                    self.slot_names.append(slot)

        return {'id': self.getID()}

    def save(self, filename=None, append=False, sort=True):
        """Save word and action dictionaries to outer files."""
        if filename:
            self.words.save(filename + '.words', sort=sort)
            self.actions.save(filename + '.actions', sort=sort)
            json.dump(self.slot_names, open(filename + '.slots', 'w'))
        else:
            self.words.save(sort=sort)
            self.actions.save(sort=sort)

    def share(self):
        shared = {}
        shares['words'] = self.words
        shares['actions'] = self.actions
        shared['opt'] = self.opt
        shared['class'] = type(self)
        return shared

    def shutdown(self):
        """Shutdown words and actions"""
        self.words.shutdown()
        self.actions.shutdown()

    def __str__(self):
        return str(self.words) + '\n' + str(self.actions)
