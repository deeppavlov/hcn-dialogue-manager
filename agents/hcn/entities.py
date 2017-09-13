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

from enum import Enum
import numpy as np


class Babi5EntityTracker():

    class EntType(Enum):
        PARTY_SIZE = auto()
        LOCATION = auto()
        CUISINE = auto()
        REST_TYPE = auto()

        def __str__(self):
            return "<{}>".format(self.name.lower())

    def __init__(self):
        # tracker entity values
        self.entities = {
                self.EntType.PARTY_SIZE : None,
                self.EntType.LOCATION : None,
                self.EntType.CUISINE : None,
                self.EntType.REST_TYPE : None,
                }

        # possible entity values
        self.all_entities = {
                self.EntType.PARTY_SIZE: frozenset(('1', '2', '3', '4', '5', '6', '7', '8', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight')),
                self.EntType.LOCATION: frozenset(('bangkok', 'beijing', 'bombay', 'hanoi', 'paris', 'rome', 'london', 'madrid', 'seoul', 'tokyo')),
                self.EntType.CUISINE: frozenset(('british', 'cantonese', 'french', 'indian', 'italian', 'japanese', 'korean', 'spanish', 'thai', 'vietnamese')),
                self.EntType.REST_TYPE: frozenset(('cheap', 'expensive', 'moderate'))
                }

    def entity2type(self, word):
        for t in self.EntType:
            if word in self.all_entities.get(t):
                return t
        return None

    def extract_entity_types(self, tokens, update=True):
        new_tokens = []
        for token in tokens:
            ent_type = self.entity2type(token)
            if update and (ent_type is not None):
                self.entities[ent_type] = token

            new_tokens.append(str(ent_type or token))

        return new_tokens

    def binary_features(self):
        return np.array( [bool(self.entities[t]) for t in self.EntType], 
               dtype=np.float32 )

# TODO: categorical entity features
    def categ_features(self):
        return []

