import sys, math
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import pyglet
from pyglet import gl
import cv2
import mss
import pygetwindow as gw

from .custom_gym_env.utils import (
    get_window_handle, up, down, left, right, stop,
    shoot_up, shoot_down, shoot_left, shoot_right, shoot_stop,
    bomb, enter
)
from .custom_gym_env.utils import getLives, getScore, read_process_memory

import time

STATEX = 100
STATEY = 100

# Reward constants
SURVIVAL_REWARD = 0.1  # Small positive reward for surviving each step
LIFE_LOST_PENALTY = -10.0  # Large negative penalty for losing a life
SCORE_REWARD_SCALE = 0.01  # Scale factor for score-based rewards
TIME_REWARD_INTERVAL = 300  # Steps between time-based rewards
TIME_REWARD = 1.0  # Reward for surviving TIME_REWARD_INTERVAL steps
FRAME_SKIP = 1  # Number of frames to skip between actions (1 = no skipping)

def noop(window_handle):
    stop(window_handle)

class GeomEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self):
        super(GeomEnv, self).__init__()
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.reward = 0.0
        self.step_reward = 0.0
        self.prev_reward = 0.0
        self.state = []
        
        # Initialize window and screen capture
        self.window_handle = get_window_handle('Geometry Wars: Retro Evolved')
        if self.window_handle == 0:
            raise RuntimeError("Could not find Geometry Wars window")
        
        self.sct = mss.mss()
        self._update_monitor_region()
        
        # Track game state
        self.prev_lives = 3  # Initial lives
        self.prev_score = 0
        self.steps_since_last_time_reward = 0
        self.frame_count = 0

        self.action_space = spaces.MultiDiscrete([3,3,3,3,2])
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATEX, STATEY, 1), dtype=np.uint8)

        self.y_move = [noop, up, down]
        self.x_move = [noop, left, right]
        self.y_shoot = [noop, shoot_up, shoot_down]
        self.x_shoot = [noop, shoot_left, shoot_right]
        self.bomb = [noop, bomb]

    def _update_monitor_region(self):
        """Update the monitor region for screen capture."""
        try:
            geom_window = gw.getWindowsWithTitle('Geometry Wars: Retro Evolved')[0]
            self.monitor = {
                'top': geom_window.top + 32,
                'left': geom_window.left + 20,
                'width': STATEX,
                'height': STATEY
            }
        except (IndexError, AttributeError) as e:
            raise RuntimeError(f"Failed to get window position: {e}")

    def _get_observation(self):
        """Capture and process the current game screen."""
        try:
            # Update monitor region in case window moved
            self._update_monitor_region()
            
            # Capture screen
            img = np.asarray(self.sct.grab(self.monitor))
            if img is None or img.size == 0:
                raise RuntimeError("Failed to capture screen")
            
            # Process image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return np.expand_dims(img, axis=-1)
        except Exception as e:
            raise RuntimeError(f"Failed to get observation: {e}")

    def _calculate_reward(self):
        """Calculate the reward for the current step."""
        current_lives = getLives()
        current_score = getScore()
        reward = 0.0

        # Survival reward
        reward += SURVIVAL_REWARD

        # Life lost penalty
        if current_lives < self.prev_lives:
            reward += LIFE_LOST_PENALTY
        self.prev_lives = current_lives

        # Score-based reward
        score_diff = current_score - self.prev_score
        reward += score_diff * SCORE_REWARD_SCALE
        self.prev_score = current_score

        # Time-based reward
        self.steps_since_last_time_reward += 1
        if self.steps_since_last_time_reward >= TIME_REWARD_INTERVAL:
            reward += TIME_REWARD
            self.steps_since_last_time_reward = 0

        return reward

    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation and info."""
        super().reset(seed=seed)
        self.reward = 0.0
        self.step_reward = 0.0
        self.prev_reward = 0.0
        self.prev_lives = 3
        self.prev_score = 0
        self.steps_since_last_time_reward = 0
        self.frame_count = 0
        time.sleep(1)
        enter(self.window_handle)
        self.state = self._get_observation()
        return self.state, {}

    def step(self, action):
        """Execute one time step within the environment."""
        self.frame_count += 1
        
        # Apply frame skipping
        if self.frame_count % FRAME_SKIP != 0:
            return self.state, 0, False, False, {}
            
        # Execute action
        self.y_move[action[0]](self.window_handle)
        self.x_move[action[1]](self.window_handle)
        self.y_shoot[action[2]](self.window_handle)
        self.x_shoot[action[3]](self.window_handle)
        self.bomb[action[4]](self.window_handle)
        
        # Get new state and calculate reward
        self.state = self._get_observation()
        reward = self._calculate_reward()
        
        # Check if game is over
        terminated = self.prev_lives <= 0
        truncated = False
        
        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        """Render the current state."""
        if mode == 'human':
            if self.viewer is None:
                from gymnasium.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            
            # Convert state to RGB for display
            img = np.repeat(self.state, 3, axis=-1)  # Convert grayscale to RGB
            self.viewer.imshow(img)
            return self.viewer.isopen
        return None

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.sct is not None:
            self.sct.close()





