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
    agent = parser.add_argument_group('HCN Arguments')
    agent.add_argument('--no-cuda', type='bool', default=False)
    agent.add_argument('--gpu', type=int, default=-1)
    agent.add_argument('--random_seed', type=int, default=123098)

    # Basics
    agent.add_argument('--pretrained-model', type=str, default=None,
                       help='Load dict/features/weights/opts/database from '
                       'the file prefix')
    agent.add_argument('--log-file', type=str, default=None)
# agent.add_argument('--tracker', required=True, choices=['babi5', 'babi6'],
#                   help='Type of entity tracker to use. Implemented only '
#                   'for dialog_babi5 and dialog_babi6.')

    # Dictionaries configuration
    agent.add_argument('--fasttext-model', type=str, default=None,
                        help='Fasttext pretrained model')
    agent.add_argument('--embedding-file', type=str, default=None,
                        help='List of used word embeddings: w e1 .. ed.')
    agent.add_argument('--template-file', type=str, default=None,
                       help='File with dataset templates if present.')

    # Model specification
    agent.add_argument('--learning-rate', type=float, default=.1)
    agent.add_argument('--epoch-num', type=int, default=1)
    agent.add_argument('--hidden-dim', type=int, default=128)
    agent.add_argument('--action-mask', type='bool', default=False,
                       help='Use action mask to put constrains on actions.')
