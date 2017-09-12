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

import sys
import copy
from parlai.core.agents import Agent

from . import config
from .model import HybridCodeNetworkModel


def class HybridCodeNetworkAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        self.id = 'HybridCodeNetworkAgent'
        self.episode_done = True
        self.n_examples = 0

        super().__init__(opt, shared)

        if shared is not None:
            self.is_shared = True
        else:
            self.is_shared = False
            self.model = HybridCodeNetworkModel(opt) 

    def observe(self, observation):
        """Receive an observation/action dict."""
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if previous observation was not an end of a dialog, restore it
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        """Return an observation/action dict based upon given observation."""
        pass

    def reset_metrics(self):
        self.model.reset_metrics()
        self.n_examples = 0

    def save(self, fname=None):
        """Save the parameters of the agent."""
        fname = fname or self.opt.get('model_file', None)
        if fname:
            sys.stderr.write("<INFO> saving model to '{}'\n".format(fname))
            self.model.save(fname)
        else:
            sys.stderr.write("<WARN> failed to save model.\n")

    def shutdown(self):
        """Final cleanup."""
        if not self.is_shared:
            if self.model is not None:
                self.model.shutdown()
            self.model = None

