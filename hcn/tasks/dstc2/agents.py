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

from parlai.core.agents import MultiTaskTeacher
from .build import build
from .teacher import DSTC2Teacher

import copy
import os


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    prefix = os.path.join(opt['datapath'], 'dstc2')
    suffix = ''
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        suffix = 'trn'
    elif dt == 'valid':
        suffix = 'val'
    elif dt ==  'test':
        suffix = 'tst'
    datafile = os.path.join(prefix, 'dstc2-{type}.jsonlist'.format(type=suffix))
    cands_datafile = os.path.join(prefix, 'dstc2-cands.txt')

    return datafile, cands_datafile


class DefaultTeacher(DSTC2Teacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        paths = _path(opt)
        opt['datafile'], opt['cands_datafile'] = paths
        super().__init__(opt, shared)

