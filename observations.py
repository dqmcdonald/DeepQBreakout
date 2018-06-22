#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 07:37:17 2018

@author: que
"""

import pickle

from collections import deque
import numpy as np
import random


import zlib

STATE_FRAMES = 2
SCREEN_HEIGHT = 84
SCREEN_WIDTH = 72

MAX_MEMORY = 20000
OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)

class Observations(object):
    """ Stores the observations made when playing Breakout"""
    
    def __init__(self):
        
        self.observations = deque(maxlen = MAX_MEMORY)
        self.last_state = None
       
        
    def __len__(self):
        return len(self.observations)
    
    def addObservation(self,screen_image, terminal, action, reward ):
        """
        Add an observation.
        screen_image is the image of the current screen
        terminal is a flag which indicates if the game is over
        action is either a one-hot array of key actions or a single keypress
        """
        
        processed_image = self.preProcess(screen_image)
        
        action_array = []
        if type(action) == int:
            # Convert to one-hot encoding
            action_array = [0 for i in range(3)]
            action_array[action-1] = 1
            action = action_array

        # first frame must be handled differently
    
        if self.last_state is None:
            self.last_state = np.stack(tuple(processed_image for _ in range(STATE_FRAMES)), axis=2)
        else:
            screen_binary = np.reshape(processed_image, (SCREEN_HEIGHT, SCREEN_WIDTH, 1))
            current_state = np.append( self.last_state[:,:, 1:], screen_binary, axis=2)
        
            self.observations.append( zlib.compress(pickle.dumps((self.last_state, action, reward, current_state,
                                                        terminal), 2), 2))
        
            self.last_state = current_state
    
    def saveToFile( self, filename):
        with open(filename, mode='wb') as f:
            pickle.dump( self.observations, f)
    
    
    def loadFromFile(self, filename):
        try:
            with open(filename, mode='rb') as f:
                self.observations = pickle.load(f)
        except FileNotFoundError:
            pass
    
    def preProcess( self, screen_image ):
        "Reduce and crop the screen image to remove any unneeded information"
        screen_image = screen_image[32:-10, 8:-8] # crop
        screen_image = screen_image[::2, ::2, 0]  # downsample by a factor of 2
        screen_image[screen_image != 0] == 1  # set everything as either black:0 or white:1
    
        return screen_image.astype(np.float)
    
    
    def getMiniBatch( self, batch_size):
        """
        Return five lists of batch_size length:
            previous_states
            actions
            rewards
            current_states
            terminal
        """
        
        
        # sample a mini-batch to train on
        mini_batch_compressed = random.sample( self.observations, batch_size)
        mini_batch = [pickle.loads(zlib.decompress(comp_items)) for comp_items in mini_batch_compressed]
         
        # get the batch variables:
        previous_states = [d[OBS_LAST_STATE_INDEX] for d in mini_batch]
        actions = [d[OBS_ACTION_INDEX] for d in mini_batch]
        rewards = [d[OBS_REWARD_INDEX] for d in mini_batch]
        current_states = [d[OBS_CURRENT_STATE_INDEX] for d in mini_batch]
        terminal = [d[OBS_TERMINAL_INDEX]for d in mini_batch]
        
        return (previous_states, actions, rewards, current_states, terminal)
        
        