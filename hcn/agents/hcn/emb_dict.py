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
import urllib.request
import fasttext


class EmbeddingsDict(object):
    """Contains word embeddings"""

    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        self.tok2emb = {}
        self.dim = 0
        self.load_items()

        self.fasttext_model = None
        if self.opt.get('fasttext_model') is not None:
            if not os.path.isfile(self.opt['fasttext_model']):
                emb_path = os.environ.get('EMBEDDINGS_URL')
                if not emb_path:
                    raise RuntimeError('No pretrained fasttext model provided')
                fname = os.path.basename(opt['fasttext_model'])
                try:
                    print('Trying to download a pretrained fasttext model'
                        ' from the repository')
                    url = urllib.parse.urljoin(emb_path, fname)
                    urllib.request.urlretrieve(url, opt['fasttext_model'])
                    print('Downloaded a fasttext model')
                except Exception as e:
                    raise RuntimeError('Looks like the `EMBEDDINGS_URL` variable'
                                    ' is set incorrectly', e)
            self.fasttext_model = fasttext.load_model(opt['fasttext_model'])
            if self.tok2emb and (self.fasttext_model.dim != self.dim):
                raise RuntimeError("Fasttext model and loaded embeddings have"
                                   " different dimension sizes.")
        else:
            print("No fasttext model provided: using loaded embeddings.")

    def add_items(self, tokens):
        if self.fasttext_model is not None:
            for token in tokens:
                if self.tok2emb.get(token) is None:
                    self.tok2emb[token] = self.fasttext_model[token]

    def __contains__(self, token):
        if self.fasttext_model and (token not in self.tok2emb):
            self.tok2emb[token] = self.fasttext_model[token]
        return token in self.tok2emb

    def __getitem__(self, token):
        if self.fasttext_model and (token not in self.tok2emb):
            self.tok2emb[token] = self.fasttext_model[token]
        return self.tok2emb.get(token, None)

    def encode(self, tokens):
        embs = [self.__getitem__(t) for t in tokens if self.__contains__(t)]
        if embs:
            return np.mean(embs, axis=0)
        return np.zeros(self.dim, dtype=np.float32)

    def save_items(self, fname):
        if self.opt.get('embedding_file') is not None:
            fname = self.opt['embedding_file']
        else:
            fname += '.emb'
        f = open(fname, 'w')
        string = '\n'.join([el[0] + ' ' + self.emb2str(el[1]) for el in self.tok2emb.items()])
        f.write(string)
        f.close()

    def emb2str(self, vec):
        string = ' '.join([str(el) for el in vec])
        return string

    def load_items(self):
        """Initialize embeddings from file."""
        fname = None
        if self.opt.get('embedding_file') is not None:
            fname = self.opt['embedding_file']
        elif self.opt.get('pretrained_model') is not None:
            fname = self.opt['pretrained_model']+'.emb'
        elif self.opt.get('model_file') is not None:
            fname = self.opt['model_file']+'.emb'

        if fname is None or not os.path.isfile(fname):
            print('There is no %s file provided. Initializing new dictionary.' % fname)
        else:
            print('Loading existing dictionary from %s.' % fname)
            with open(fname, 'r') as f:
                for line in f:
                    values = line.strip().rsplit(sep=' ')
                    word = values[0]
                    weights = np.asarray(values[1:], dtype=np.float32)
                    self.tok2emb[word] = weights
                    if not self.dim:
                        self.dim = len(weights)

