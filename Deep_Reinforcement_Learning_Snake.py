#!/usr/bin/env python
# coding: utf-8

import logging
import math
import numpy as np
#from PIL import ImageGrab #if windows or os X
import pyscreenshot as ImageGrab  #if linux
from PIL import Image
import cv2 #opencv
import io
import time
#from matplotlib import pyplot as plt
#plt.rcParams['figure.figsize'] = (30, 30)
#import seaborn as sns
import pandas as pd
import numpy as np
from random import randint
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
#%matplotlib inline 
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard
from collections import deque
import random
import pickle
import json

game_url = "./snake.html"
chrome_driver_path = "../chromedriver.exe"
loss_file_path = "./data/loss_df.csv"
actions_file_path = "./data/actions_df.csv"
scores_file_path = "./data/scores_df.csv"
time_file_path = "./data/time_df.csv"
log_file_path = "./data/game_logs.log"

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename=log_file_path, level=logging.INFO)

if os.path.isfile(loss_file_path):
    loss_df = pd.read_csv(loss_file_path)
else:
    loss_df = pd.DataFrame(columns =['loss'])
    f = open(loss_file_path, "w+")
    loss_df.to_csv(loss_file_path, index=False)

if os.path.isfile(scores_file_path):
    scores_df = pd.read_csv(scores_file_path)
else:
    scores_df = pd.DataFrame(columns = ['scores'])
    open(scores_file_path, "x")
    scores_df.to_csv(scores_file_path, index=False)

if os.path.isfile(actions_file_path):
    actions_df = pd.read_csv(actions_file_path)
else:
    actions_df = pd.DataFrame(columns = ['left', 'right', 'up', 'down'])
    open(actions_file_path, "x")
    actions_df.to_csv(actions_file_path, index=False)

if os.path.isfile(time_file_path):
    time_df = pd.read_csv(time_file_path)
else:
    time_df = pd.DataFrame(columns = ['time'])
    open(time_file_path, "x")
    time_df.to_csv(time_file_path, index=False)


#sudo apt-get install chromium-chromedriver

img_rows, img_cols = 64,64
SPEED = 360
WIDTH = 8
HEIGHT = 8
ACTIONS = 4 
GAMMA = 0.99 
OBSERVATION = 65.
EXPLORE = 70
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 1
REPLAY_MEMORY = 50000
BATCH = 64
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_channels = 4

class Game:
    def __init__(self,speed, width, height):
        self.speed = speed
        self.width = width
        self.height = height
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
        # self._driver = webdriver.Chrome(executable_path = "chromedriver.exe",chrome_options=chrome_options)
        self._driver.set_window_position(x=-10,y=0)
        self._driver.set_window_size(img_rows*5+50,img_cols*5+50)
        self._driver.get("file://"+os.path.abspath(game_url))
        self._driver.execute_script("Init.instance_.speed="+str(self.speed))
        print(self.width)
        self._driver.execute_script("Init.instance_.width="+str(self.width))
        self._driver.execute_script("Init.instance_.height="+str(self.height))
    def get_crashed(self):
        return self._driver.execute_script("return Init.instance_.crashed")
    def get_playing(self):
        return self._driver.execute_script("return Init.instance_.playing")
    def restart(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ENTER)
        
        time.sleep(0.25)# no actions are possible 
                        # for 0.25 sec after game starts, 
                        # skip learning at this time and make the model wait
    def press_enter(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ENTER)
    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    def press_down(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
    def press_left(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_LEFT)
    def press_right(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_RIGHT)
    def get_time(self):
        time = self._driver.execute_script("return Init.instance_.time")
        return int(time)
    def get_score(self):
        score = self._driver.execute_script("return Init.instance_.score")
        return int(score)
    def get_just_eaten(self):
        just_eaten = self._driver.execute_script("return Init.instance_.just_eaten")
        return just_eaten
    def get_distance(self):
        distance = self._driver.execute_script("return Init.instance_.distance")
        return distance
    def get_food_loc(self):
        foodx = self._driver.execute_script("return Init.instance_.foodx")
        foody = self._driver.execute_script("return Init.instance_.foody")
        return foodx, foody
    def get_head_loc(self):
        headx = self._driver.execute_script("return Init.instance_.headx")
        heady = self._driver.execute_script("return Init.instance_.heady")
        return headx, heady
    def end(self):
        self._driver.close()


class Agent:
    def __init__(self,game):
        self._game = game
        self.start()
        time.sleep(.5)
    def is_running(self):
        return self._game.get_playing()
    def is_crashed(self):
        return self._game.get_crashed()
    def is_just_eaten(self):
        return self._game.get_just_eaten()
    def up(self):
        self._game.press_up()
    def start(self):
        self._game.press_enter()
    def down(self):
        self._game.press_down()
    def left(self):
        self._game.press_left()
    def right(self):
        self._game.press_right()


class State:
    def __init__(self,agent,game):
        self._agent = agent
        self._game = game
        self._display = show_img()
        self._display.__next__() 
        self.prev_dist = 0
    def get_state(self,actions):
        actions_df.loc[len(actions_df)] = list(actions)
        score = self._game.get_score() 
        time = self._game.get_time()
        distance = self._game.get_distance()
        reward = 0
        is_over = False
        if actions[0] == 1:
            self._agent.left()
        elif actions[1] == 1:
            self._agent.right()
        elif actions[2] == 1:
            self._agent.up()
        elif actions[3] == 1:
            self._agent.down()
        image = grab_screen() 
        self._display.send(image)
        # is_eaten = self._game.get_just_eaten()
        reward = 0.1*(self.prev_dist - distance)
        self.prev_dist = distance
        if distance == 0:
            reward = 1
        if self._agent.is_crashed():
            scores_df.loc[len(scores_df)] = score
            time_df.loc[len(time_df)] = time
            reward = -1
            self._game.restart()
            is_over = True
        reward = round(reward,4)
        # print('reward: ' +str(reward))
        return image, reward, is_over


def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def grab_screen(_driver = None):
    x = 80
    y = 240
    screen =  np.array(ImageGrab.grab(bbox=(x+10,y,x+WIDTH*13,y+HEIGHT*13))) #bbox = region of interset on the entire screen
    image = process_img(screen)
    return image


def process_img(image):
    image = cv2.resize(image, (img_rows,img_cols))
    image = cv2.Canny(image, threshold1 = 100, threshold2 = 200)
    return  image


def show_img(graphs = False):
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
        imS = cv2.resize(screen, (200, 100)) 
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break


def init_cache():
    save_obj(INITIAL_EPSILON,"epsilon")
    t = 0
    save_obj(t,"time")
    D = deque()
    save_obj(D,"D")


def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, (7, 7), strides=(4, 4), padding='same',input_shape=(img_cols,img_rows,img_channels)))  #20*40*4
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model


# buildmodel().summary()


def trainNetwork(model,game_state,observe=False):
    last_time = time.time()
    D = deque()
    do_nothing = np.zeros(ACTIONS)
#     do_nothing[0] = 1 #0 =>do nothing, 1=>left, 2=>right, 3=>up, 4=>down
    
    x_t, r_0, terminal = game_state.get_state(do_nothing)
#     print('x_t: {}'.format(x_t.shape))
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
#     print('s_t: {}'.format(s_t.shape))
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
#     print('s_t reshaped: {}'.format(s_t.shape))
    initial_state = s_t 
    model.save_weights("model_final.h5")

    if observe :
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model_final.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
        model.load_weights("model_final.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
    t=0
    t = load_obj("time") # resume from the previous time step stored in file system
    print(t)
    while (True): #endless running
        
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if  random.random() <= epsilon:
                #print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else: # predict the output
                q = model.predict(s_t)
                max_Q = np.argmax(q)
                action_index = max_Q 
                a_t[action_index] = 1
                
        #Reduced the epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE 

        #run the selected action and observed next state and reward
        x_t1, r_t, terminal = game_state.get_state(a_t)
        #print('reward: {}'.format(r_t))
        #print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
#         print('x_t1: {}'.format(x_t1.shape))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
#         print('s_t1: {}'.format(s_t1))
        
        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE: 
            
            #sample a minibatch to train on
            print(len(D), BATCH)
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((inputs.shape[0], ACTIONS))

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]    # 2D stack of images
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]   #reward at state_t due to action_t
                state_t1 = minibatch[i][3]   #next state
                terminal = minibatch[i][4]   #wheather the agent died or survided due the action
                

                inputs[i:i + 1] = state_t

                targets[i] = model.predict(state_t)  # predicted q values
                Q_sa = model.predict(state_t1)      # predict q values for next step
                
                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
        else:
            time.sleep(0.10)
        s_t = initial_state if terminal else s_t1 
        t = t + 1
        
        if t % 1000 == 0:
            print("TimeStep: ", t, "Save model", end='')
            model.save_weights("model_final.h5", overwrite=True)
            save_obj(D,"D")
            save_obj(t,"time")
            save_obj(epsilon,"epsilon")
            loss_df.to_csv(loss_file_path,index=False)
            #print(scores_df.tail())
            scores_df.to_csv(scores_file_path,index=False)
            time_df.to_csv(time_file_path,index=False)
            actions_df.to_csv(actions_file_path,index=False)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        # logging.info('TIMESTEP ' + str(t) +
        #               '| STATE ' + str(state) +
        #               '| EPSILON ' + str(epsilon) +
        #               '| ACTION ' + str(action_index) +
        #               '| Q_MAX ' + str(np.max(Q_sa)) +
        #               '| REWARD ' + str(r_t) +
        #               '| Loss ' + str(loss)
        #             )
        # print('.', end='')
        print("TIMESTEP", t, "/ STATE", state, "/ REWARD", r_t,"/ EPSILON", epsilon, "/ ACTION", action_index, "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")


def playGame(observe=False):
    game = Game(speed=SPEED, width=WIDTH, height=HEIGHT)
    snake = Agent(game)
    game_state = State(snake,game)
    model = buildmodel()
    try:
        trainNetwork(model,game_state)
    except StopIteration:
        game.end()


init_cache()
playGame(observe=False);


# supervised_frames = np.load("training_data_final_working.npy")
# frame = supervised_frames[0][0]
# action_index = supervised_frames[0][1]
# plt.imshow(frame)
# print('Action taken at this frame : Action index = {} i.e. jump'.format(str(action_index)))


# supervised_actions = []

# for frame in supervised_frames:
#     supervised_actions.append(frame[1])


# fig, axs = plt.subplots(ncols=1,nrows =2,figsize=(15,15))
# sns.distplot(supervised_actions,ax=axs[0])
# axs[1].set_title('AI gameplay distribution')
# axs[0].set_title('Human gameplay distribution')
# actions_df = pd.read_csv("./data/actions_df.csv")
# sns.distplot(actions_df,ax=axs[1])

