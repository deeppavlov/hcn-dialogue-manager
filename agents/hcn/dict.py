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

import spacy

from parlai.core.dict import DictionaryAgent

from . import config
from .utils import normalize_text


NLP = spacy.load('en')


class ActionDictionaryAgent(DictionaryAgent):
    """Override DictionaryAgent to user spaCy tokenizer, ignore labels
    and count also actions.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--pretrained_words', type='bool', default=True,
            help='User only words found in provided embedding_file'
            )

    def __init__(self, *args, **kwargs):
        self.id = self.__class__.__name__
        super().__init__(*args, **kwargs)

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

    def tokenize(self, text, **kwargs):
        tokens = NLP.tokenizer(text)
        return (t.text for t in tokens)

    def add_to_dict(self, tokens):
        """Build dictionary from the list of provided tokens.
        Only add words contained in self.embedding_words, if not None.
        """
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
        """Add only words passed in the 'text' field of the observation to 
        the dictionary.
        """
        for text in self.observation.get('text'):
            if text:
                self.add_to_dict(self.tokenize(text))
        return {'id': self.getID()}

