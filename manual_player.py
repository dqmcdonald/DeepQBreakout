import curses

import numpy as np
import gym

MANUAL_FILE = "manual.obs"
BALL_UPDATE= 28000  # Update the ball every 28000 cycles

from  observations import Observations


obs = Observations()

obs.loadFromFile(MANUAL_FILE)
scores_per_game = []
num_games = 0
high_score=0



def main(win):
    global num_games
    global scores_per_game
    global high_score
    env = gym.make("Breakout-v0")
    env.reset()
    env.render()
    running = False
    
    current_game_score = 0
    win.nodelay(True)
    key=""
    count = 0
    win.addstr("Press G to begin")
    while 1:          
        action = 0

        try:                 
           key = win.getkey()         
           win.clear()                
           if str(key) == "Q" or str(key) == 'q':
               env.close()
               break
           if str(key) == "KEY_LEFT":
                action = 3
           if str(key) == "KEY_RIGHT":
                action = 2
           if str(key) == "G" or str(key) == "g":
               running = True
           
                
        except Exception as e:
           # No input   
           pass      
       
        if running:
            count += 1
            if( count > BALL_UPDATE ):
                # Every N cycles update the ball
                count = 0
                action = 1
            if action > 0:
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
                    obs.resetLastState()
         

curses.wrapper(main)
obs.saveToFile(MANUAL_FILE)
print("There are {} observations".format(len(obs)))
print("Average score per game = {:.2f} from {} games".format(np.mean(np.array(scores_per_game)), num_games))
print("High score = {}".format(high_score))
