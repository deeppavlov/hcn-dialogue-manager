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

import unicodedata
import numpy as np


# ------------------------------------------------------------------------------
# Data/model utilities.
# ------------------------------------------------------------------------------


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def is_api_call(text):
    return text.strip().startswith('api_call')


def is_api_answer(text):
    return (not is_silence(text)) and text.strip().endswith('<SILENCE>')


def is_null_api_answer(text):
    return text.strip().startswith('api_call no result')


def is_silence(text):
    return text.strip() == '<SILENCE>'


def filter_service_words(tokens):
    return filter(lambda t: '_' not in t, tokens)

# ------------------------------------------------------------------------------
# Babi5&Babi6 specific utilities.
# ------------------------------------------------------------------------------


def babi6_dirty_fix(text):
    """Fix some inconsistencies in DSTC2 data preparation."""
    return text.replace('the cow pizza kitchen and bar',
                        'the_cow_pizza_kitchen')\
        .replace('the good luck chinese food takeaway', 'the_good_luck')\
        .replace('the river bar steakhouse and grill', 'the_river_bar')\
        .replace(' Fen Ditton', '')\
        .replace('ask is', 'R_name is')\
        .replace('ask serves', 'R_name serves')\
        .replace('01223 323737', 'R_phone')\
        .replace('C.B 2, 1 U.F', 'R_post_code')\
        .replace('C.B 1, 3 N.F', 'R_post_code')\
        .replace('C.B 2, 1 D.P', 'R_post_code')\
        .replace('C.B 4, 3 L.E', 'R_post_code')\
        .replace('108 Regent Street City Centre', 'R_address')\
        .replace('17 Magdalene Street City Centre', 'R_address')\
        .replace('529 Newmarket Road', 'R_address')\
        .replace('7 Milton Road Chesterton', 'R_address')


def iter_api_response(text):
    info = {}
    for ln in text.split('\n'):
        tokens = ln.split()
        if is_silence(ln):
            yield info
        if (len(tokens) != 3):
            return
        rest, prop, value = tokens
        value = int(value) if value.isdecimal() else value
        if not info:
            info['R_name'] = rest
        if info['R_name'] == rest:
            info[prop] = value
        else:
            yield info
            info = {'R_name': rest, prop: value}
