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
# Action type handle utilities.
# -----------------------------------------------------------------------------

def extract_babi5_template(tokens):
    template = []
    for token in tokens:
        if 'resto_' in token: 
            if 'phone' in token:
                template.append('<info_phone>')
            elif 'address' in token:
                template.append('<info_address>')
            else:
                template.append('<restaurant>')
        else:
            template.append(token)
    return ' '.join(template)

