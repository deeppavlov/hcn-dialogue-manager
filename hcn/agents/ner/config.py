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


def add_cmdline_args(parser):
    # Runtime environment
    agent = parser.add_argument_group('NER Arguments')
    agent.add_argument('--ner-dict-filepath',
                       default='build/nerpa/dict.txt',
                       help='Path to the nerpa dictionary for Corpus class instance.')
    agent.add_argument('--ner-model-filepath',
                       default='build/nerpa/ner_model.ckpt',
                       help='Path to the NER tensorflow model.')
    agent.add_argument('--ner-params-filepath',
                       default='build/nerpa/params.json',
                       help='Path to the parameters of the NER model')
    agent.add_argument('--ner-slot-vals-filepath',
                       default='build/nerpa/slot_vals.json',
                       help='Path to the slot values dictionary (with variations of the slot values)')
