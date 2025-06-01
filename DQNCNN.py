import h5py
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.models import load_model

import os
import datetime


### For gpu training on WSL
# import tensorflow.compat.v1 as tf 
# tf.enable_eager_execution(tf.ConfigProto(log_device_placement=True)) 

class DQNCNNAgent:
    def __init__(self, environment, trained_model=None):
        # Initialize constant
        self.environment = environment
        self.obs_size = environment.observation_space.shape  # Now includes channel dimension
        self.action_size = np.prod(environment.action_space.nvec)  # Total number of actions
        self.consecutive_episodes = 100

        # Hyperparameters of the training
        self.learning_rate = 0.0005
        self.gamma = 0.99  # discount factor
        self.replay_memory = 50000
        self.replay_size = 128

        # Initialize neural network model
        if trained_model:
            self.model = self.load_model(filename=trained_model)
        else:
            self.model = self.build_model()

        # Exploration/exploitations parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.001
        self.episode_b4_replay = 4

        # Define variable
        self.storage = deque(maxlen=self.replay_memory)
        self.sum_reward, self.rewards_lst = 0.0, []


    def build_model(self):
        """Build the CNN model for Deep-Q learning."""
        model = Sequential([
            # First convolutional layer
            Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=self.obs_size),
            # Second convolutional layer
            Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'),
            # Third convolutional layer
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
            # Flatten the output
            Flatten(),
            # Dense layers
            Dense(512, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    def store(self, state, action, reward, next_state, done):
        """Save experience to replay memory."""
        # Convert action tuple to single integer index
        action_idx = np.ravel_multi_index(action, self.environment.action_space.nvec)
        self.storage.append((state, action_idx, reward, next_state, done))

        
    def action(self, state, reward, done, episode, training=True):
        """Select an action using epsilon-greedy policy."""
        # Update cumulative reward
        self.sum_reward += reward

        # Episode ends
        if done:
            self.rewards_lst.append(self.sum_reward)
            avg_reward = np.mean(self.rewards_lst[-self.consecutive_episodes:])
            print('Episode %4d, Reward: %5d, Average rewards %5d' % (episode, self.sum_reward, avg_reward))
            self.sum_reward = 0.0
            self.epsilon = max(self.epsilon_decay * self.epsilon, self.epsilon_min)
            return -1

        # Episode not ends: return next action
        else:
            # Add batch dimension to state
            state = np.expand_dims(state, axis=0)
            
            # Train agent
            if training:
                if episode >= self.episode_b4_replay:
                    self.replay()
                    if np.random.random() < self.epsilon:
                        action = self.environment.action_space.sample()
                    else:
                        act_values = self.model.predict(state, verbose=0)
                        action_idx = np.argmax(act_values[0])
                        # Convert single index back to action tuple
                        action = np.unravel_index(action_idx, self.environment.action_space.nvec)
                else:
                    action = self.environment.action_space.sample()
            # Run trained agent
            else:
                act_values = self.model.predict(state, verbose=0)
                action_idx = np.argmax(act_values[0])
                action = np.unravel_index(action_idx, self.environment.action_space.nvec)

            return action


    def replay(self):
        """Train the model using experience replay."""
        if len(self.storage) < self.replay_size:
            return

        minibatch_idx = np.random.permutation(len(self.storage))[:self.replay_size]

        states = np.array([self.storage[i][0] for i in minibatch_idx])
        actions = np.array([self.storage[i][1] for i in minibatch_idx])
        rewards = np.array([self.storage[i][2] for i in minibatch_idx])
        next_states = np.array([self.storage[i][3] for i in minibatch_idx])
        dones = np.array([self.storage[i][4] for i in minibatch_idx])

        # Get current Q-values and next Q-values
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        # Compute target Q-values
        target_q_values = current_q_values.copy()
        for i in range(self.replay_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the model
        self.model.fit(states, target_q_values, batch_size=self.replay_size, epochs=1, verbose=0)


    def action_to_index(self, actions):
        """Convert action tuples to single integer indices."""
        # For MultiDiscrete action space [3,3,3,3,2]
        # Convert each action component to one-hot encoding
        y_move = tf.one_hot(actions[:, 0], 3)
        x_move = tf.one_hot(actions[:, 1], 3)
        y_shoot = tf.one_hot(actions[:, 2], 3)
        x_shoot = tf.one_hot(actions[:, 3], 3)
        bomb = tf.one_hot(actions[:, 4], 2)
        
        # Concatenate all one-hot encodings
        return tf.concat([y_move, x_move, y_shoot, x_shoot, bomb], axis=1)

    def save(self, filename):
        """Save the model to a file."""
        self.model.save(filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load the model from a file."""
        self.model = tf.keras.models.load_model(filename)
        print(f"Model loaded from {filename}")


    def load_model(self, filename):
        """Load the model from a file."""
        return load_model(filename)