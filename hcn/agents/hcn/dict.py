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

from .utils import filter_service_words, normalize_text


NLP = spacy.load('en')


class SpacyDictionaryAgent(DictionaryAgent):
    """Override DictionaryAgent to use Spacy tokenizer"""

    def __init__(self, opt, shared):
        super().__init__(self.opt, shared)
        self.word_tok = None

    def tokenize(self, text, **kwargs):
        """Tokenize with spacy, placing service words as individual tokens."""
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

