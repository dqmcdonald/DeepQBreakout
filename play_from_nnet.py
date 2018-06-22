#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:57:18 2018

@author: que

Play Breakout from a previously trained neural net

"""

import tensorflow as tf
import gym
import nnetwork
import observations
import argparse
import os
import numpy as np


ACTIONS_COUNT = 3


parser = argparse.ArgumentParser()
parser.add_argument("name", help="The name for the observations and checkpoint files");
parser.add_argument("-n", "--ngames", type=int, 
   dest="num_games", help="The number of games to be played", default=200)
parser.add_argument("-r", "--random_prob", type=float, 
   dest="random_prob", help="The fraction of moves to be random", default=0.0)
parser.add_argument("-s", "--save_obs", type=str, 
dest="save_obs", help="The file name to save oversvations to", default="")
args = parser.parse_args()


obs = observations.Observations()
obsfilename = args.name + ".obs"
obs.loadFromFile(obsfilename)


scores_per_game = []
num_games = 0
high_score=0


chkfilename = "./" + args.name 
print("Loading observations from {}".format(obsfilename))


session = tf.Session()
input_layer, output_layer, action, target, train_operation = nnetwork.create_network()

session.run(tf.global_variables_initializer())
 
saver = tf.train.Saver(max_to_keep=1)

if os.path.exists(chkfilename + ".meta"):
    print("Restoring from checkpoint file {}".format(chkfilename))
    checkpoint = tf.train.get_checkpoint_state(chkfilename)
    if checkpoint:
        saver.restore( session, checkpoint.model_checkpoint_path)
else:
    raise ValueError("Checkpoint not found")



env = gym.make("Breakout-v0")
env.reset()
env.render()
    
    
current_game_score = 0
 # Set the first action to do nothing
last_action = np.zeros(ACTIONS_COUNT)
last_action[1] = 1  

while num_games < args.num_games:          
    
       
    env.render()
    screen_image, reward, terminal, info = env.step(nnetwork.key_presses_from_action(last_action))
    current_game_score += reward
    obs.addObservation( screen_image, terminal, last_action, reward)
    last_state = obs.getLastState()
    last_action = nnetwork.choose_next_action(session, input_layer,output_layer, last_state, random_prob=args.random_prob)
    
    if terminal:
        num_games += 1
        scores_per_game.append(current_game_score)
        if current_game_score > high_score:
            high_score = current_game_score
        current_game_score = 0
        env.reset()
         

env.close()

if args.save_obs:
    save_file_name = args.save_obs + ".obs"
    print("Saving observations as {}".format(save_file_name))
    obs.saveToFile(save_file_name)

print("Average score per game = {:.2f} from {} games".format(np.mean(np.array(scores_per_game)), num_games))
print("High score = {}".format(high_score))

