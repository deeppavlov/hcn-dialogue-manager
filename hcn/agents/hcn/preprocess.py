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

from parlai.core.dict import DictionaryAgent, Agent

from .utils import normalize_text
from .utils import is_api_answer, is_null_api_answer, iter_api_response
from .dict import WordDictionaryAgent, ActionDictionaryAgent
from .entities import Babi5EntityTracker, Babi6EntityTracker
from .database import DatabaseSimulator


class HCNPreprocessAgent(Agent):
    """Contains WordDictionaryAgent and ActionDictionaryAgent."""

    @staticmethod
    def add_cmdline_args(argparser):
        WordDictionaryAgent.add_cmdline_args(argparser)
        ActionDictionaryAgent.add_cmdline_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        self.id = self.__class__.__name__

        # database
        self.database = None
        if not shared and opt.get('model_file'):
            database_file = opt['model_file'] + '.db'
            self.database = DatabaseSimulator(database_file)

        # initialize entity tracker
        self.tracker = None
        if opt['tracker'] == 'babi5':
            self.tracker = Babi5EntityTracker
        elif opt['tracker'] == 'babi6':
            self.tracker = Babi6EntityTracker

        # intialize action dictionary
        self.actions = ActionDictionaryAgent(opt, shared)

        # initialize word dictionary
        self.words = WordDictionaryAgent(opt, shared)

    def update_database(self, text):
        if not is_null_api_answer(text):
            self.database.insert_many(list(iter_api_response(text)))

    def act(self):
        """
            - Add words passed in the 'text' field of the observation to
        the dictionary,
            - extract action templates from all 'label_candidates' once
            - update database from api responses in 'text' field.
        """
        # if utterance is an api response, update database
        text = self.observation.get('text')
        if text and is_api_answer(text):
                self.update_database(text)

        # add to word dict
        self.words.observe(self.observation)
        self.words.act()

        # add to action dict
        self.actions.observe(self.observation)
        self.actions.act()

        return {'id': self.getID()}

    def save(self, filename=None, append=False, sort=True):
        """Save word and action dictionaries to outer files."""
        if filename:
            self.words.save(filename + '.words', sort=sort)
            self.actions.save(filename + '.actions', sort=sort)
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
