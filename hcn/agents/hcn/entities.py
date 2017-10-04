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
import copy


class EntityTracker():

    # types of entities
    class EntType(Enum):
        pass
    # entity types to track
    tracked_types = []
    # possible entity values, if no entity -- any value is acceptable
    possible_values = {}

    @property
    def num_features(self):
        return len(self.tracked_types)

    def __init__(self):
        self.restart()

    def restart(self):
        self.tracked = {}

    @property
    def entities(self):
        return {str(t): v for t, v in self.tracked.items()}

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
            if ent_type in self.tracked_types:
                self.tracked[ent_type] = token
            new_tokens.append(str(ent_type or token))
        return new_tokens

    def binary_features(self):
        return np.array([(t in self.tracked) for t in self.tracked_types],
                        dtype=np.float32)

    def categ_features(self):
        return np.array([self.possible_values[t].index(self.tracked[t]) + 1
                         if t in self.tracked else 0
                         for t in self.tracked_types],
                        dtype=np.float32)

    def fill_entities(self, text):
        for ent_type, value in self.tracked.items():
            text = text.replace(str(ent_type), value)
        return text


class Babi5EntityTracker(EntityTracker):

    class EntType(Enum):
        R_number = 1
        R_location = 2
        R_cuisine = 3
        R_price = 4
        R_phone = 5
        R_address = 6
        R_name = 7

        def __str__(self):
            return "{}".format(self.name)

    tracked_types = [
        EntType.R_number, EntType.R_location, EntType.R_cuisine,
        EntType.R_price
    ]

    possible_values = {
        EntType.R_number: [
            '1', '2', '3', '4', '5', '6', '7', '8', 'one', 'two', 'three',
            'four', 'five', 'six', 'seven', 'eight'
        ],
        EntType.R_location: [
            'bangkok', 'beijing', 'bombay', 'hanoi', 'paris', 'rome', 'london',
            'madrid', 'seoul', 'tokyo'
        ],
        EntType.R_cuisine: [
            'british', 'cantonese', 'french', 'indian', 'italian', 'japanese',
            'korean', 'spanish', 'thai', 'vietnamese'
        ],
        EntType.R_price: [
            'cheap', 'expensive', 'moderate'
        ]
    }

    @classmethod
    def entity2type(cls, word):
        for t in cls.tracked_types:
            if word in cls.possible_values.get(t):
                return t
        if '_phone' in word:
            return cls.EntType.R_phone
        if '_address' in word:
            return cls.EntType.R_address
        if 'resto_' in word:
            return cls.EntType.R_name
        return None


class Babi6EntityTracker(EntityTracker):

    class EntType(Enum):
        R_name = 1
        R_location = 2
        R_cuisine = 3
        R_price = 4
        R_post_code = 5
        R_phone = 6
        R_address = 7

        def __str__(self):
            return "{}".format(self.name)

    tracked_types = [
        EntType.R_name, EntType.R_location, EntType.R_cuisine, EntType.R_price
    ]

    possible_values = {
        EntType.R_name: [
            'nandos_city_centre', 'de_luca_cucina_and_bar', 'hakka', 'venue',
            'restaurant_alimentum', 'golden_wok', 'efes_restaurant',
            'restaurant_two_two', 'meze_bar_restaurant', 'the_hotpot',
            'zizzi_cambridge', 'graffiti', 'pizza_hut_city_centre',
            'frankie_and_bennys', 'the_lucky_star', 'taj_tandoori',
            'the_missing_sock', 'bangkok_city', 'caffe_uno',
            'maharajah_tandoori_restaurant', 'tandoori_palace', 'india_house',
            'the_golden_curry', 'the_varsity_restaurant',
            'pizza_hut_cherry_hinton', 'the_river_bar', 'riverside_brasserie',
            'city_stop_restaurant', 'bloomsbury_restaurant', 'golden_house',
            'saigon_city', 'mahal_of_cambridge', 'pizza_express', 'sala_thong',
            'anatolia', 'ugly_duckling', 'grafton_hotel_restaurant', 'cotto',
            'restaurant_one_seven', 'shanghai_family_restaurant', 'rajmahal',
            'eraina', 'the_cow_pizza_kitchen', 'the_good_luck', 'royal_spice'
            'yu_garden', 'midsummer_house_restaurant', 'thanh_binh',
            'michaelhouse_cafe', 'la_margherita', 'rice_house',
            'the_slug_and_lettuce', 'the_nirala', 'don_pasquale_pizzeria',
            'kymmoy', 'pipasha_restaurant', 'prezzo', 'la_tasca',
            'stazione_restaurant_and_coffee_bar', 'panahar',
            'hotel_du_vin_and_bistro', 'ali_baba', 'chiquito_restaurant_bar',
            'curry_prince', 'pizza_hut_fen_ditton', 'saint_johns_chop_house',
            'curry_king', 'j_restaurant', 'dojo_noodle_bar',
            'the_copper_kettle', 'lan_hong_house', 'curry_garden',
            'da_vinci_pizzeria', 'kohinoor', 'the_cambridge_chop_house',
            'charlie_chan', 'nandos', 'loch_fyne', 'meghna',
            'yippee_noodle_bar', 'cocum', 'the_gardenia', 'royal_standard',
            'jinling_noodle_bar', 'travellers_rest', 'bedouin',
            'gourmet_burger_kitchen', 'la_raza', 'clowns_cafe', 'little_seoul',
            'hk_fusion', 'la_mimosa', 'sitar_tandoori', 'the_gandhi',
            'backstreet_bistro', 'saffron_brasserie',
            'cambridge_lodge_restaurant', 'wagamama', 'rice_boat',
            'sesame_restaurant_and_bar', 'darrys_cookhouse_and_wine_shop',
            'curry_queen', 'the_oak_bistro', 'cote', 'shiraz_restaurant',
            'galleria', 'fitzbillies_restaurant', 'peking_restaurant'
        ],
        EntType.R_location: [
            'south', 'north', 'centre', 'west', 'east'
        ],
        EntType.R_cuisine: [
            'british', 'portuguese', 'mediterranean', 'asian_oriental',
            'african', 'japanese', 'seafood', 'international', 'gastropub',
            'korean', 'italian', 'bistro', 'thai', 'spanish', 'french',
            'indian', 'UNK', 'european', 'turkish', 'vietnamese', 'chinese',
            'lebanese', 'north_american', 'northern_european', 'fusion',
            'mexican', 'modern_european', 'afghan', 'australasian',
            'australian', 'austrian', 'barbeque', 'basque', 'belgian',
            'brazilian', 'canapes', 'cantonese', 'caribbean', 'catalan',
            'christmas', 'corsica', 'creative', 'crossover', 'cuban',
            'danish', 'english', 'eritrean', 'german', 'greek', 'halal',
            'hungarian', 'indonesian', 'irish', 'jamaican', 'kosher',
            'malaysian', 'middle_eastern', 'moroccan', 'panasian', 'persian',
            'polish', 'polynesian', 'romanian', 'russian', 'scandinavian',
            'scottish', 'singaporean', 'steakhouse', 'swedish', 'swiss',
            'the_americas', 'traditional', 'tuscan', 'unusual', 'vegetarian',
            'venetian', 'welsh', 'world'
        ],
        EntType.R_price: [
            'cheap', 'expensive', 'moderate'
        ]
    }

    @classmethod
    def entity2type(cls, word):
        for t in cls.tracked_types:
            if word in cls.possible_values.get(t):
                return t
        if '_phone' in word:
            return cls.EntType.R_phone
        if '_address' in word:
            return cls.EntType.R_address
        if '_post_code' in word:
            return cls.EntType.R_post_code
        return None

    @classmethod
    def extract_entity_types(cls, tokens):
        if tokens[:6] == ['Would', 'you', 'like', 'something', 'in', 'the']:
            return tokens
        new_tokens = []
        for token in tokens:
            ent_type = cls.entity2type(token)
            new_tokens.append(str(ent_type or token))
        return new_tokens
