import gym
import random

import numpy as np
import keras
from statistics import mean, median
from collections import Counter

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense
LR = 1e-3

env = gym.make('CartPole-v0')
env.reset()


def showGame(nr = 10):

    for _ in range(nr):
        print (_)
        env.reset()
        
        while True:
            env.render()
            
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            if done:  break
    


def saveGoodGames(nr=10000):
    observations = []
    actions = []
    minReward = 70

    for i in range(nr):
#        print (_)
        env.reset()
        action = env.action_space.sample()
        
        obserVationList = []
        actionList = []
        score = 0
        while True:
#            env.render()
            
            observation, reward, done, info = env.step(action)
            action = env.action_space.sample()
            obserVationList.append(observation)
            if action == 1:

                actionList.append([0,1] )
            elif action == 0:

                actionList.append([1,0])
            score += reward
            if done:  break

#        print (score,  actionList )
        if score > minReward:
            observations.extend(obserVationList)
            actions.extend(actionList)
    observations = np.array(observations)
    actions = np.array(actions)
    return observations, actions


def trainModell(observations=None, actions= None):

    if not observations:
        observations = np.load('observations.npy')
    if not actions:
        actions = np.load('actions.npy')


    model = Sequential()
    model.add(Dense(64, input_dim=4, activation='relu'))
    model.add(Dense(128,  activation='relu'))
    model.add(Dense(256,  activation='relu'))
    model.add(Dense(256,  activation='relu'))
    model.add(Dense(2,  activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    model.fit(observations, actions, epochs=10)
    model.save('basic.h5')


def playGames(nr=10000, ai=None):

    ai = load_model('basic.h5')

    observations = []
    actions = []
    minReward = 70
    scores=0
    scores = []

    for i in range(nr):
        env.reset()
        action = env.action_space.sample()
        
        obserVationList = []
        actionList = []
        score=0
        while True:
#            env.render()
            
            observation, reward, done, info = env.step(action)
            action = np.argmax(ai.predict(observation.reshape(1,4)))
            obserVationList.append(observation)
            if action == 1:

                actionList.append([0,1] )
            elif action == 0:

                actionList.append([1,0])
            score += 1
#            score += reward
            if done:  break

        print (score  )
        scores.append(score)
        if score > minReward:
            observations.extend(obserVationList)
            actions.extend(actionList)
    observations = np.array(observations)
    actions = np.array(actions)
    print (np.mean(scores))
    return observations, actions

obs, acts = saveGoodGames()

print (obs.shape, acts.shape)
np.save('observations.npy', obs)
np.save('actions.npy', acts)


trainModell()
playGames(30)
