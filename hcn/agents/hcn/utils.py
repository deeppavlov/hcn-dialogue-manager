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
# Babi5 specific utilities.
# ------------------------------------------------------------------------------

def extract_babi5_template(tokens):
    template = []
    for token in tokens:
        if 'resto_' in token: 
            if 'phone' in token:
                template.append('R_phone')
            elif 'address' in token:
                template.append('R_address')
            else:
                template.append('resto_')
        else:
            template.append(token)
    return template


def iter_babi5_api_response(text):
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
            info['resto_'] = rest
        if info['resto_'] == rest: 
            info[prop] = value
        else:
            yield info
            info = {'resto_': rest, prop: value}

#TODO: filling of restaurant info
def is_babi5_restaurant(word):
    return word.startswith('resto_')

