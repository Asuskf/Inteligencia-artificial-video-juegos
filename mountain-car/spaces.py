# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 06:10:36 2020

@author: david
"""

import gym
from gym.spaces import *
import sys

def print_spaces(space):
    print(space)
    
    if isinstance(space, Box):
        print("\n Cota inferior: ", space.low)
        print("\n Cota superior: ", space.high)

if __name__ == "__main__":
    environment = gym.make(sys.argv[1])
    print("Espacio de estados:")
    print_spaces(environment.observation_space)
    print("Espacio de acciones:")
    print_spaces(environment.action_space)
    
    try:
        print("Descripción de las acciones: ", environment.unwrapped.get_action_meaning())
    except AttributeError:
        pass