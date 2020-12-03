# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import gym
import sys

environment = gym.make("MountainCar-v0")
MAX_NUM_EPISODES = 1000


for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = environment.reset()
    total_reward = 0.0
    step = 0
    
    while not done:
        environment.render()
        action = environment.action_space.sample()
        next_state, reward, done, info = environment.step(action)
        total_reward += reward
        step += 1
        obs = next_state
    
    print("\n Episodio número {} finalizado con {} iteraciones. Recompesa final = {}".format(episode, step+1, total_reward))

environment.close()    

