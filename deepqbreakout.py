
# 

import pickle
import random
from collections import deque

import tensorflow as tf
import gym
import numpy as np
import os

import zlib

resume = True

CHECKPOINT_PATH = 'deep_q_breakout_cp'

SCREEN_HEIGHT = 84
SCREEN_WIDTH = 72


ACTIONS_COUNT = 3
FUTURE_REWARDS_DISCOUNT = 0.99
OBSERVATION_STEPS = 100000.0  # time steps to observe before training
EXPLORE_STEPS = 2000000 # frames over which to anneal epsilon
INITIAL_RANDOM_ACTION_PROB = 1.0 # starting chance of an action being random
FINAL_RANDOM_ACTION_PROB = 0.05  # final chance of a move being random
MEMORY_SIZE = 400000
MINI_BATCH_SIZE = 128  # size of minibatches
STATE_FRAMES = 2      # number of frames to store in the state
OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
SAVE_EVERY_X_STEPS = 2000
LEARN_RATE = 1e-4
STORE_SCORES_LEN = 100
verbose_logging = False
PRINT_EVERY_X_STEPS = 5000

training=False

# Reduce and crop the screen image to remove any unneeded information
def pre_process( screen_image ):
    screen_image = screen_image[32:-10, 8:-8] # crop
    screen_image = screen_image[::2, ::2, 0]  # downsample by a factor of 2
    screen_image[screen_image != 0] == 1  # set everything as either black:0 or white:1
    
    return screen_image.astype(np.float)

def create_network():
    
    CONVOLUTIONS_LAYER_1 = 32
    CONVOLUTIONS_LAYER_2 = 64
    CONVOLUTIONS_LAYER_3 = 64
    FLAT_HIDDEN_NODES = 512
    FLAT_SIZE = 11*9*CONVOLUTIONS_LAYER_3
    WINDOW_SIZE_1 = 8
    WINDOW_SIZE_2 = 4
    WINDOW_SIZE_3 = 3
    
    input_layer = tf.placeholder("float", [None,SCREEN_HEIGHT, SCREEN_WIDTH, STATE_FRAMES])
    
    convolution_weights_1 = tf.Variable(tf.truncated_normal([WINDOW_SIZE_1,WINDOW_SIZE_1,STATE_FRAMES, CONVOLUTIONS_LAYER_1], stddev=0.01))
    convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[CONVOLUTIONS_LAYER_1]))
    
    hidden_convolution_layer_1 = tf.nn.relu( tf.nn.conv2d( input_layer, convolution_weights_1, strides=[1,4,4,1], 
                                                         padding = "SAME") + convolution_bias_1)
    
    
    convolution_weights_2 = tf.Variable(tf.truncated_normal([WINDOW_SIZE_2,WINDOW_SIZE_2,CONVOLUTIONS_LAYER_1, CONVOLUTIONS_LAYER_2], stddev=0.01))
    convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[CONVOLUTIONS_LAYER_2]))
    
    hidden_convolution_layer_2 = tf.nn.relu( tf.nn.conv2d( hidden_convolution_layer_1, convolution_weights_2, strides=[1,2,2,1], 
                                                         padding = "SAME") + convolution_bias_2)
    
    
    convolution_weights_3 = tf.Variable(tf.truncated_normal([WINDOW_SIZE_3,WINDOW_SIZE_3,CONVOLUTIONS_LAYER_2, CONVOLUTIONS_LAYER_3], stddev=0.01))
    convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[CONVOLUTIONS_LAYER_2]))
    
    hidden_convolution_layer_3 = tf.nn.relu( tf.nn.conv2d( hidden_convolution_layer_2, convolution_weights_3, strides=[1,1,1,1], 
                                                         padding = "SAME") + convolution_bias_3)

    hidden_convolution_layer_3_flat = tf.reshape(hidden_convolution_layer_3, [-1,FLAT_SIZE])
    
    feed_forward_weights_1 = tf.Variable( tf.truncated_normal([FLAT_SIZE, FLAT_HIDDEN_NODES],stddev=0.01))
    
    feed_forward_bias_1 = tf.Variable( tf.constant( 0.01, shape=[FLAT_HIDDEN_NODES]))
    
    final_hidden_activations = tf.nn.relu( tf.matmul(hidden_convolution_layer_3_flat, feed_forward_weights_1) +
                                        feed_forward_bias_1)
    
    feed_forward_weights_2 = tf.Variable(tf.truncated_normal([FLAT_HIDDEN_NODES, ACTIONS_COUNT],stddev=0.01))
    
    feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[ACTIONS_COUNT]))
    
    output_layer = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2
    
    return input_layer, output_layer


def choose_next_action(state):
    
    new_action = np.zeros([ACTIONS_COUNT])
    
    if random.random() <= probability_of_random_action:
        # choose a random action:
        action_index = random.randrange(ACTIONS_COUNT)
    else:
        # Choose an action given our last state:
        readout_t = session.run(output_layer, feed_dict={input_layer :[state]})[0]
        
        if verbose_logging:
            print("Action Q-values are %s" % readout_t)
        
        action_index = np.argmax(readout_t)
        
    new_action[action_index] = 1
    return new_action


def key_presses_from_action( action_set ):
    if action_set[0] == 1:
        return 1
    if action_set[1] == 1:
        return 2
    if action_set[2] == 1:
        return 3
    raise ValueError("Unexpected action")
    

 
    
def train():
    global training
    if not training:
        print("Starting training")
        training = True
    
    
    # sample a mini-batch to train on
    mini_batch_compressed = random.sample( observations, MINI_BATCH_SIZE)
    mini_batch = [pickle.loads(zlib.decompress(comp_items)) for comp_items in mini_batch_compressed]
    
    # get the batch variables:
    previous_states = [d[OBS_LAST_STATE_INDEX] for d in mini_batch]
    actions = [d[OBS_ACTION_INDEX] for d in mini_batch]
    rewards = [d[OBS_REWARD_INDEX] for d in mini_batch]
    current_states = [d[OBS_CURRENT_STATE_INDEX] for d in mini_batch]
    agents_expected_reward = []
    
    # this gives us the agents expected reward for each action we might take:
    agents_reward_per_action = session.run(output_layer, feed_dict = {input_layer:current_states})
    for i in range(len(mini_batch)):
        if mini_batch[i][OBS_TERMINAL_INDEX]:
            # this was a terninal frame so there is no future reward:
            agents_expected_reward.append(rewards[i])
        else:
            agents_expected_reward.append( rewards[i] + FUTURE_REWARDS_DISCOUNT*np.max(agents_reward_per_action[i]))
            
    
    # learn that these actions in these states lead to this reward:
    session.run( train_operation, feed_dict = {input_layer: previous_states,
                                              action: actions,
                                              target: agents_expected_reward})
    
    # save checkpoints for later:
    if time % SAVE_EVERY_X_STEPS == 0:
        saver.save( session, CHECKPOINT_PATH+'/network', global_step = time)
    
    
    

session = tf.Session()
input_layer, output_layer = create_network()

action = tf.placeholder("float", [None,ACTIONS_COUNT])
target = tf.placeholder("float", [None])

readout_action = tf.reduce_sum(tf.multiply(output_layer, action), reduction_indices =1)

cost = tf.reduce_mean( tf.square( target - readout_action))
train_operation = tf.train.AdamOptimizer(LEARN_RATE).minimize(cost)

observations = deque(maxlen=MEMORY_SIZE)
last_scores = deque(maxlen=STORE_SCORES_LEN)

# Set the first action to do nothing
last_action = np.zeros(ACTIONS_COUNT)
last_action[1] = 1

last_state = None
probability_of_random_action = INITIAL_RANDOM_ACTION_PROB
time = 0


session.run(tf.global_variables_initializer())

saver = tf.train.Saver()

if not os.path.exists(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)
    
if resume:
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_PATH)
    if checkpoint:
        saver.restore( session, checkpoint.model_checkpoint_path)
        


env = gym.make("Breakout-v0")
observation = env.reset()
reward = 0
score_per_game = 0

while True:
    
    env.render()
    
    observation, reward, terminal, info = env.step(key_presses_from_action(last_action))
    score_per_game += reward
    
    screen_binary = pre_process(observation)
    

    # first frame must be handled differently
    
    if last_state is None:
        last_state = np.stack(tuple(screen_binary for _ in range(STATE_FRAMES)), axis=2)
    else:
        screen_binary = np.reshape(screen_binary, (SCREEN_HEIGHT, SCREEN_WIDTH, 1))
        current_state = np.append( last_state[:,:, 1:], screen_binary, axis=2)
        
        observations.append( zlib.compress(pickle.dumps((last_state, last_action, reward, current_state,
                                                        terminal), 2), 2))
        
        # only train if done observing:
        if len(observations) > OBSERVATION_STEPS:
            train()
            time += 1
        
        if terminal:
            last_scores.append(score_per_game)
            score_per_game = 0
            env.reset()
            last_state = None
        else:
            last_state = current_state
            last_action = choose_next_action(last_state)
            
            
        
           # gradually reduce the probability of a random action
        if probability_of_random_action > FINAL_RANDOM_ACTION_PROB             and len(observations) > OBSERVATION_STEPS:
            probability_of_random_action -=             (INITIAL_RANDOM_ACTION_PROB - FINAL_RANDOM_ACTION_PROB) / EXPLORE_STEPS

        if time > 0 and time % PRINT_EVERY_X_STEPS == 0:
            print("Time {} random action probability: {} reward: {} scores differential {}".format(
                time, probability_of_random_action, reward, np.mean(last_scores)))

    
