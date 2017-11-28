"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed inder the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import spacy
import copy
import re

from parlai.core.dict import DictionaryAgent

from . import entities
from .utils import filter_service_words, babi6_dirty_fix, normalize_text


NLP = spacy.load('en')


class SpacyDictionaryAgent(DictionaryAgent):
    """Override DictionaryAgent to use Spacy tokenizer and
    use preprocessing tricks of dialog_babi5 and dialog_babi6.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        argparser.add_argument(
            '--tracker', required=False, choices=['babi5', 'babi6'],
            help='Type of entity tracker to use. Implemented only '
                 'for dialog_babi5 and dialog_babi6.')
        return argparser

    def __init__(self, opt, shared):
        super().__init__(self.opt, shared)
        self.word_tok = None

    def tokenize(self, text, **kwargs):
        """Tokenize with spacy, placing service words as individual tokens."""
        if self.opt['tracker'] == 'babi6':
            text = babi6_dirty_fix(text)
        text = text.replace('<SILENCE>', '_SILENCE_')

        return [t.text for t in NLP.tokenizer(text)]

    def detokenize(self, tokens):
        """
        Detokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `detokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.
        """
        text = ' '.join(tokens)
        step0 = text.replace('. . .',  '...')
        step1 = step0.replace("`` ", '"').replace(" ''", '"')
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" n't", "n't")\
            .replace(" nt", "nt").replace("can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        return step6.strip()


class WordDictionaryAgent(SpacyDictionaryAgent):
    """Override SpacyDictionaryAgent to ignore labels and words not in
    pretrained_words.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        SpacyDictionaryAgent.add_cmdline_args(argparser)
        dictionary = argparser.add_argument_group('Word Dictionary Arguments')
        dictionary.add_argument(
            '--pretrained-words', type='bool', default=False,
            help='User only words found in provided embedding_file'
            )
        return dictionary

    def __init__(self, opt, shared=None):
        self.id = self.__class__.__name__

        # initialize DictionaryAgent
        self.opt = copy.deepcopy(opt)
        if self.opt.get('dict_file') is not None:
            self.opt['dict_file'] = self.opt['dict_file'] + '.words'
        elif self.opt.get('pretrained_model') is not None:
            self.opt['dict_file'] = self.opt['pretrained_model'] + '.dict.words'
        elif self.opt.get('model_file') is not None:
            self.opt['dict_file'] = self.opt['model_file'] + '.dict.words'
        super().__init__(self.opt, shared)

        # index words in embedding file
        if self.opt['pretrained_words'] is not None\
           and self.opt.get('embedding_file') is not None:
            print('[ Indexing words with embeddings... ]')
            self.embedding_words = set()
            with open(self.opt['embedding_file']) as f:
                for line in f:
                    w = normalize_text(line.strip().split(' ', 1)[0])
                    self.embedding_words.add(w)
            print('[ Num words in set = % d ]' % len(self.embedding_words))
        else:
            self.embedding_words = None

    def add_to_dict(self, tokens):
        """Build dictionary from the list of provided tokens.
        Only add words contained in self.embedding_words, if not None.
        """
# TODO: ?add normalization of a token?
        for token in tokens:
            if self.embedding_words and (token not in self.embedding_words):
                continue
            self.freq[token] += 1
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token

    def act(self):
        """Add words passed in the 'text' field of the observation to
        the dictionary.
        """
        text = self.observation.get('text')
        if text:
            self.add_to_dict(filter_service_words(self.tokenize(text)))
        return {'id': self.getID()}


class ActionDictionaryAgent(SpacyDictionaryAgent):
    """Override SpacyDictionaryAgent to count actions."""

    @staticmethod
    def add_cmdline_args(argparser):
        dictionary = argparser.add_argument_group('Action Dictionary'
                                                  ' Arguments')
        dictionary.add_argument(
            '--action-file',
            help='if set, the dictionary will automatically save to this path'
                 ' during shutdown')
        return dictionary

    def __init__(self, opt, shared=None):
        self.id = self.__class__.__name__

        # initialize DictionaryAgent
        self.opt = {
            'tracker': opt.get('tracker'),
            'dict_max_ngram_size': -1,
            'dict_minfreq': 0,
            'dict_nulltoken': None,
            'dict_endtoken': None,
            'dict_unktoken': None,
            'dict_starttoken': None,
            'dict_language': SpacyDictionaryAgent.default_lang
        }
        if opt.get('action_file') is not None:
            self.opt['dict_file'] = opt['action_file']
        elif opt.get('dict_file') is not None:
            self.opt['dict_file'] = opt['dict_file'] + '.actions'
        elif opt.get('pretrained_model') is not None:
            self.opt['dict_file'] = opt['pretrained_model'] + '.dict.actions'
        elif opt.get('model_file') is not None:
            self.opt['dict_file'] = opt['model_file'] + '.dict.actions'
        super().__init__(self.opt, shared)
        '''if shared:
            self.freq = shared.get('freq', {})
            self.tok2ind = shared.get('tok2ind', {})
            self.ind2tok = shared.get('ind2tok', {})
        else:
            self.freq = defaultdict(int)
            self.tok2ind = {}
            self.ind2tok = {}'''

        # entity tracker class methods
        self.tracker = None
        if self.opt['tracker'] == 'babi5':
            self.tracker = entities.Babi5EntityTracker
        elif self.opt['tracker'] == 'babi6':
            self.tracker = entities.Babi6EntityTracker

        # properties
        self.label_candidates = False

    def get_template(self, tokens):
        if self.tracker:
            tokens = self.tracker.extract_entity_types(tokens)
        return self.detokenize(tokens)

    def act(self):
        """Extract action templates from 'label_candidates' once."""
        if not self.label_candidates:
            self.label_candidates = True
            for text in self.observation.get('label_candidates', ()):
                if text:
                    tokens = self.tokenize(text)
                    self.add_to_dict([self.get_template(tokens)])

        return {'id': self.getID()}
