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

from enum import Enum, auto
import numpy as np


class Babi5EntityTracker():

    class EntType(Enum):
        R_number = auto()
        R_location = auto()
        R_cuisine = auto()
        R_price = auto()

        def __str__(self):
            return "{}".format(self.name)

    # possible entity values
    all_entities = {
            EntType.R_number: frozenset(('1', '2', '3', '4', '5', '6', '7', '8', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight')),
            EntType.R_location: frozenset(('bangkok', 'beijing', 'bombay', 'hanoi', 'paris', 'rome', 'london', 'madrid', 'seoul', 'tokyo')),
            EntType.R_cuisine: frozenset(('british', 'cantonese', 'french', 'indian', 'italian', 'japanese', 'korean', 'spanish', 'thai', 'vietnamese')),
            EntType.R_price: frozenset(('cheap', 'expensive', 'moderate'))
            }

    num_features = len(EntType)

    def __init__(self):
        self.restart()

    def restart(self):
        self.entities = {}

    @classmethod
    def entity2type(cls, word):
        for t in cls.EntType:
            if word in cls.all_entities.get(t):
                return t
        return None

    @classmethod
    def extract_entity_types(cls, tokens):
        new_tokens = []
        for token in tokens:
            ent_type = cls.entity2type(token)
            new_tokens.append(str(ent_type or token))
        return new_tokens

    def update_entities(self, tokens):
        new_tokens = []
        for token in tokens:
            ent_type = self.entity2type(token)
            if ent_type is not None:
                self.entities[ent_type] = token
            new_tokens.append(str(ent_type or token))
        return new_tokens

    def binary_features(self):
        return np.array( [(t in self.entities) for t in self.EntType], 
               dtype=np.float32 )

# TODO: categorical entity features
    def categ_features(self):
        return []

    def fill_entities(self, text):
        for ent_type, value in self.entities.items():
            text = text.replace(str(ent_type), value)
        return text

