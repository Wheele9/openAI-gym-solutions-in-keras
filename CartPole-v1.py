import gym
import random

import numpy as np
import keras
from statistics import mean, median
from collections import Counter

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense
LR = 1e-3

env = gym.make('CartPole-v1')
env.reset()


def saveGoodGames(nr=10000):
    observations = []
    actions = []
    minReward = 70

    for i in range(nr):
        env.reset()
        action = env.action_space.sample()
        
        obserVationList = []
        actionList = []
        score = 0
        while True:
            
            observation, reward, done, info = env.step(action)
            action = env.action_space.sample()
            obserVationList.append(observation)
            if action == 1:

                actionList.append([0,1] )
            elif action == 0:

                actionList.append([1,0])
            score += reward
            if done:  break

        if score > minReward:
            observations.extend(obserVationList)
            actions.extend(actionList)
    observations = np.array(observations)
    actions = np.array(actions)
    return observations, actions

def trainModell(modelName, observations=None, actions= None, ):

    if  observations== None:
        observations = np.load('observations.npy')
    if actions == None:
        actions = np.load('actions.npy')


    model = Sequential()
    model.add(Dense(64, input_dim=4, activation='relu'))
    model.add(Dense(128,  activation='relu'))
    model.add(Dense(256,  activation='relu'))
    model.add(Dense(256,  activation='relu'))
    model.add(Dense(64,  activation='relu'))
    model.add(Dense(2,  activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    model.fit(observations, actions, epochs=10)
    model.save('{}.h5'.format(modelName))
    return model


def playGames( ai,nr,  minScore=300):
    

    observations = []
    actions = []
    scores=0
    scores = []

    for i in range(nr):
        if i%50==0: print ('step {}'.format(i))
        env.reset()
        action = env.action_space.sample()
        
        obserVationList = []
        actionList = []
        score=0
        while True:
            env.render()
            
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

#        print (score  )
        scores.append(score)
        if score > minScore:
            observations.extend(obserVationList)
            actions.extend(actionList)
    observations = np.array(observations)
    actions = np.array(actions)
    print ('mean: ', np.mean(scores))
    return observations, actions

#obs, acts = saveGoodGames()
##
##print (obs.shape, acts.shape)
#np.save('observationsR1.npy', obs)
#np.save('actionsR1.npy', acts)
##
##
#firstModel = trainModell( 'v1',obs, acts)
#obs, acts = playGames(firstModel, 1000, 300)

##print (obs.shape, acts.shape)
#np.save('observationsR2.npy', obs)
#np.save('actionsR2.npy', acts)
#secondModel = trainModell('v2',obs, acts)
secondModel = load_model('v2.h5')
playGames(secondModel, 10)





















