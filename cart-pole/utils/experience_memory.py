# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 06:04:56 2020

@author: david
"""
import random
from collections import namedtuple

Experience = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs', 'done'])

class ExperienceMemory(object):
    def __init__(self, capacity = int(1e6)):
        self.capacity = capacity
        self.memory_idx = 0
        self.memory = []
    
        
    def sample(self, batch_size):
        assert batch_size <= self.get_size()
        return random.sample(self.memory, batch_size)
    
    def get_size(self):
        return len(self.memory)
    
    def store(self, exp):
        self.memory.insert(self.memory_idx % self.capacity, exp)
        self.memory_idx += 1
        