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

LEARN_RATE = 1e-4
ACTIONS_COUNT = 3
MINI_BATCH_SIZE = 128
SAVE_CHECK_EVERY_X = 2000  # Save checkpoint every 10000 steps
PRINT_EVERY_X = 500

parser = argparse.ArgumentParser()
parser.add_argument("name", help="The name for the observations and checkpoint files");
parser.add_argument("-n", "--nsteps", type=int, 
   dest="nsteps", help="The number of steps of training to be performed", default=20000)
args = parser.parse_args()



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



