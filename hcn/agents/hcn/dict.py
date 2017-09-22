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
import spacy
import string

from parlai.core.dict import DictionaryAgent

from .entities import Babi5EntityTracker 
from .database import DatabaseSimulator
from .utils import normalize_text, is_silence, is_api_answer, is_null_api_answer
from .utils import filter_service_words
from .utils import extract_babi5_template, iter_babi5_api_response


NLP = spacy.load('en')


class ActionDictionaryAgent(DictionaryAgent):
    """Override DictionaryAgent to user spaCy tokenizer, ignore labels
    and count also actions.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--pretrained-words', type='bool', default=True,
            help='User only words found in provided embedding_file'
            )

    def __init__(self, opt, shared=None):
        self.id = self.__class__.__name__
        super().__init__(opt, shared)

        # index words in embedding file
        if self.opt['pretrained_words'] and self.opt.get('embedding_file'):
            print('[ Indexing words with embeddings... ]')
            self.embedding_words = set()
            with open(self.opt['embedding_file']) as f:
                for line in f:
                    w = normalize_text(line.strip().split(' ', 1)[0])
                    self.embedding_words.add(w)
            print('[ Num words in set = % d ]' % len(self.embedding_words))
        else:
            self.embedding_words = None

        # action templates
        self.action_templates = []
        if not shared and opt.get('dict_file'):
            action_file = opt['dict_file'] + '.actions'
            if os.path.isfile(action_file):
                # load pre-existing action templates
                self.load_actions(action_file)

        # database
        self.database = None
        if not shared and opt.get('dict_file'):
            database_file = opt['dict_file'] + '.db'
            self.database = DatabaseSimulator(database_file)

        # entity tracker
        self.tracker = Babi5EntityTracker

    def tokenize(self, text, **kwargs):
        """Tokenize with spacy, placing service words as individual tokens."""
        tokens = [t.text for t in NLP.tokenizer(text)]

        new_tokens = []
        opening_i = None
        for i, t in enumerate(tokens):
            if t == '<':
                opening_i = i
            elif opening_i is not None:
                if t == '>':
                    bracket_expr = '<' + ' '.join(tokens[opening_i+1:i]) + '>'
                    new_tokens.append(bracket_expr)
                    opening_i = None
            else:
                new_tokens.append(t)
        return new_tokens

    def detokenize(self, tokens):
        return "".join([" " + i \
                if not i.startswith("'") and i not in string.punctuation else i \
                for i in tokens]).strip()

    def add_to_dict(self, tokens):
        """Build dictionary from the list of provided tokens.
        Only add words contained in self.embedding_words, if not None.
        """
# TODO: ?add normalization of a token?
        for token in tokens:
            if self.embedding_words is not None and \
                token not in self.embedding_words:
                continue
            self.freq[token] += 1
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token

    def act(self):
        """
            - Add words passed in the 'text' field of the observation to 
        the dictionary,
            - extract action templates from all 'label_candidates' once
            - update database from api responses in 'text' field.
        """
        # if utterance is an api response, update database
        # if utterance is not <SILENCE> or an api_call response, add to dict
        text = self.observation.get('text')
        if text:
            if is_api_answer(text): 
                self.update_database(text)
            elif not is_silence(text):
                self.add_to_dict(filter_service_words(self.tokenize(text)))

        # if action_templates not extracted, extract them
        if not self.action_templates:
            actions = set()
            for cand in self.observation.get('label_candidates'):
                if cand:
                    tokens = self.tracker.extract_entity_types(self.tokenize(cand))
                    actions.add(self.detokenize(extract_babi5_template(tokens)))
            self.action_templates = sorted(actions)

        return {'id': self.getID()}

    def update_database(self, text):
        if not is_null_api_answer(text):
            results = sorted(list(iter_babi5_api_response(text)), 
                    key=lambda r: r['R_rating'],
                    reverse=True)
            self.database.insert_many(results)
            return results
        return []

    def get_action_id(self, tokens):
        action = self.detokenize(extract_babi5_template(tokens))
        return self.action_templates.index(action)

    def get_action_by_id(self, action_id):
        return self.action_templates[action_id]

    def load_actions(self, filename):
        """Load pre-existing action templates."""
        print('Dictionary: loading action templates from {}'.format(filename))
        with open(filename) as read:
            for line in read:
                self.action_templates.append(line.strip())
        print('[ num action templates =  %d ]' % len(self.action_templates))

    def save(self, filename=None, append=False, sort=True):
        """Save dictionary and actions to outer files."""
        super().save(filename, append=append, sort=sort)
        self.save_actions(filename + '.actions', append=append)

    def save_actions(self, filename=None, append=False):
        """Save action templates to file.
        Templates are separated by ends of lines.
        
        If ``append`` (default ``False``) is set to ``True``, appends instead of rewriting.
        """
        if self.opt.get('dict_file'):
            filename = filename or self.opt['dict_file'] + '.actions'
        if filename is None:
            print('Dictionary: action templates aren\'t saved: filename not specified.')
        else:
            print('Dictionary: saving action templates to {}'.format(filename))
            with open(filename, 'a' if append else 'w') as write:
                for action in self.action_templates:
                    write.write('{}\n'.format(action))

    def share(self):
        shared = {}
        shared['freq'] = self.freq
        shared['tok2ind'] = self.tok2ind
        shared['ind2tok'] = self.ind2tok
        shared['action_templates'] = self.action_templates
        shared['opt'] = self.opt
        shared['class'] = type(self)
        return shared

    def shutdown(self):
        """Save dictionary and actions on shutdown if ``save_path`` is set."""
        if hasattr(self, 'save_path'):
            self.save(self.save_path)

    def __str__(self):
        return str(self.freq) + '\n' + str(self.action_templates)
