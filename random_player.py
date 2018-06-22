#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:27:55 2018

@author: que
"""


import numpy as np
import gym
import random

RANDOM_FILE = "random.obs"

from  observations import Observations


obs = Observations()

obs.loadFromFile(RANDOM_FILE)
scores_per_game = []
num_games = 0
high_score=0




env = gym.make("Breakout-v0")
env.reset()
env.render()
    
    
current_game_score = 0
   

while len(obs) < 20000:          
    action = random.choice([1,2,3])

       
    env.render()
    screen_image, reward, terminal, info = env.step(action)
    current_game_score += reward
    obs.addObservation( screen_image, terminal, action, reward)
    if terminal:
        num_games += 1
        scores_per_game.append(current_game_score)
        if current_game_score > high_score:
            high_score = current_game_score
        current_game_score = 0
        env.reset()
         

env.close()
obs.saveToFile(RANDOM_FILE)
print("There are {} observations".format(len(obs)))
print("Average score per game = {:.2f} from {} games".format(np.mean(np.array(scores_per_game)), num_games))
print("High score = {}".format(high_score))
