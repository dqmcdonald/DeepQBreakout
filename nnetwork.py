
# 


import random

import tensorflow as tf

import numpy as np








SCREEN_HEIGHT = 84
SCREEN_WIDTH = 72


ACTIONS_COUNT = 3
FUTURE_REWARDS_DISCOUNT = 0.99

MINI_BATCH_SIZE = 128  # size of minibatches
STATE_FRAMES = 2      # number of frames to store in the state
OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
SAVE_EVERY_X_STEPS = 2000
LEARN_RATE = 1e-4
STORE_SCORES_LEN = 100
verbose_logging = False
PRINT_EVERY_X_STEPS = 5000



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
    
    
    action = tf.placeholder("float", [None,ACTIONS_COUNT])
    target = tf.placeholder("float", [None])

    readout_action = tf.reduce_sum(tf.multiply(output_layer, action), reduction_indices =1)

    cost = tf.reduce_mean( tf.square( target - readout_action))
    train_operation = tf.train.AdamOptimizer(LEARN_RATE).minimize(cost)
    
    return input_layer, output_layer, action, target, train_operation


def choose_next_action(session, input_layer, output_layer, state, random_prob = 0.0):
    
    new_action = np.zeros([ACTIONS_COUNT])
    
    if random.random() <= random_prob:
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
    

 
    
def train(session,input_layer, output_layer, target, train_operation, action, observations, batch_size):
    
    
    (previous_states, actions, rewards, current_states, terminal) = observations.getMiniBatch(batch_size)
    
    agents_expected_reward = []
    
    # this gives us the agents expected reward for each action we might take:
    agents_reward_per_action = session.run(output_layer, feed_dict = {input_layer:current_states})
    for i in range(len(terminal)):
        if terminal[i]:
            # this was a terninal frame so there is no future reward:
            agents_expected_reward.append(rewards[i])
        else:
            agents_expected_reward.append( rewards[i] + FUTURE_REWARDS_DISCOUNT*np.max(agents_reward_per_action[i]))
            
    
    # learn that these actions in these states lead to this reward:
    session.run( train_operation, feed_dict = {input_layer: previous_states,
                                              action: actions,
                                              target: agents_expected_reward})
    
