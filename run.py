import numpy as np
import gym
import cv2
import time
import os
import sys
sys.path.insert(1, './utils')
from ctypes import wintypes

from random import seed
from random import randint
seed(1)

import os
import ctypes
import tempfile
import datetime

from tqdm import trange

from gym.utils.read_memory import getLives, getScore, read_process_memory

from random_agent import RandomAgent
from DQNCNN import DQNCNNAgent

import tensorflow as tf

def train(environment, model_name=None, key=None):
    tdir = tempfile.mkdtemp()
    env = gym.make(environment)
    env = gym.wrappers.Monitor(env, tdir, force=True)
    agent = DQNCNNAgent(env)
    env.seed(0)
    EPISODES = 10
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/geom/' + current_time
    total_rewards = np.empty(EPISODES)

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    for episode in trange(EPISODES):
        state = env.reset()
        action = agent.action(state, 0.0, False, episode)
        summary_writer = tf.summary.create_file_writer(log_dir)
        total_rewards[episode] = 0.0
        avg_rewards = total_rewards[max(0, episode - 100):(episode + 1)].mean()

        while True:
            next_state, reward, done, _ = env.step(action)
            total_rewards[episode] += reward
            
            agent.store(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                break
                
            action = agent.action(state, reward, done, episode)
            
            if episode % 10 == 0:  # Print stats every 10 episodes
                print("Lives", getLives(), "Score", getScore(), "reward", reward, "action", action)

        score = getScore()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_rewards[episode], step=episode)
            tf.summary.scalar('game score', score, step=episode)
            tf.summary.text('action', str(action), step=episode)

        if model_name and (episode == EPISODES - 1 or episode % 10 == 0):
            agent.save_model(filename=model_name)

    env.close()

if __name__ == "__main__":
    environment = 'GeomWars-v2'
    api_key = ""
    my_model = environment + 'test1000.h5'
    train(environment=environment, key=api_key, model_name=my_model)

