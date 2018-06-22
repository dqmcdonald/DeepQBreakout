#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:01:23 2018

@author: que

Train the neural network based on previous observations:
    
    
"""

import tensorflow as tf
import nnetwork
import observations
import argparse
import os
import time

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


obsfilename = args.name + ".obs"
chkfilename = "./" + args.name 
print("Loading observations from {}".format(obsfilename))
obs = observations.Observations()
obs.loadFromFile(args.name + ".obs")


print("There are {} observations".format(len(obs)))


session = tf.Session()
input_layer, output_layer, action, target, train_operation = nnetwork.create_network()



    


session.run(tf.global_variables_initializer())
 
saver = tf.train.Saver(max_to_keep=1)

if os.path.exists(chkfilename + ".meta"):
    print("Restoring from checkpoint file {}".format(chkfilename))
    checkpoint = tf.train.get_checkpoint_state(chkfilename)
    if checkpoint:
        saver.restore( session, checkpoint.model_checkpoint_path)


step = 0

start_time = time.time()
while step < args.nsteps:
    
    step += 1
    nnetwork.train(session,input_layer, output_layer, target, train_operation, action, obs, MINI_BATCH_SIZE)
   
    
    if step % SAVE_CHECK_EVERY_X == 0:
        saved_file = saver.save(session, chkfilename)
        print("Saved file checkpoint: " + saved_file)

    if step % PRINT_EVERY_X == 0:
        print("Completed training step {:5d} out of a total {:5d} in {:3d} seconds".format(step, args.nsteps, time.time()-start_time))
        start_time = time.time()