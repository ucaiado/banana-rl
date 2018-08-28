#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement the Q-network used by the agents


@author: udacity, ucaiado

Created on 10/23/2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Begin help functions
'''


'''
End help functions
'''


class QNetwork(nn.Module):
    '''Actor (Policy) Model.'''

    def __init__(self, state_size, action_size, seed):
        '''
        Initialize parameters and build model.

        :param state_size: int. Dimension of each state
        :param action_size: int. Dimension of each action
        :param seed: int. Random seed
        '''
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        '''Build a network that maps state -> action values.'''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
