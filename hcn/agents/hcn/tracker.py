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

import numpy as np


class DefaultTracker():

    def __init__(self, slot_names):
        self.slot_names = slot_names
        self.history = []

    @property
    def state_size(self):
        return len(self.slot_names)

    def reset_state(self):
        self.history = []

    def get_slots(self):
        lasts = {}
        for slot, value in self.history:
            lasts[slot] = value
        return lasts

    def binary_features(self):
        feats = np.zeros(self.state_size, dtype=np.float32)
        lasts = self.get_slots()
        for i, slot in enumerate(self.slot_names):
            if slot in lasts:
                feats[i] = 1.
        return feats

    def diff_features(self, slots):
        feats = np.zeros(self.state_size, dtype=np.float32)
        curr_slots = self.get_slots()
        for i, slot in enumerate(self.slot_names):
            if curr_slots.get(slot) != slots.get(slot):
                feats[i] = 1.
        return feats

    def update_slots(self, slots):
        if type(slots) == list:
            self.history.extend(slots)
        elif type(slots) == dict:
            for slot, value in slots.items():
                self.history.append((slot, value))

