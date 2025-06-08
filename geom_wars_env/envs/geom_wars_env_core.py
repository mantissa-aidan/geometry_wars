print("[geom_wars_env_core.py] TOP OF FILE REACHED") # New debug print
import sys, math
import numpy as np
import os # For creating debug directory

import gymnasium as gym
from gymnasium import spaces

import pyglet
from pyglet import gl
import cv2
import mss
import pygetwindow as gw

# Corrected relative import paths
from ..utils.window_input import (
    get_window_handle, up, down, left, right, stop, # Movement
    shoot_up, shoot_down, shoot_left, shoot_right, # Directional shooting
    shoot_stop, shoot_y_stop, shoot_x_stop,      # Stop functions for shooting
    bomb, enter,                                # Other actions
    InvalidWindowHandleError                    # Custom exception
)
from ..utils.read_memory import getLives, getScore, read_process_memory

import time

# Define new state dimensions
STATEX = 128
STATEY = 128

# Reward constants
SURVIVAL_REWARD = 0.1  
LIFE_LOST_PENALTY = -10.0  
SCORE_REWARD_SCALE = 0.1  
TIME_REWARD_INTERVAL = 300  
TIME_REWARD = 1.0  
FRAME_SKIP = 1  
NEW_HIGH_SCORE_REWARD = 250.0 # New constant for high score bonus
DEFAULT_BOMB_COOLDOWN_STEPS = 1800 # Approx 1 minute if 1 step = 1 frame and game is 30fps

# New no-op specifically for the bomb action component
def bomb_specific_noop(window_handle):
    """A no-op that does nothing, for when the agent chooses not to bomb."""
    pass

class GeomEnv(gym.Env):
    metadata = {
        'render_modes': ['human'],
    }

    def __init__(self, debug_mode=False, debug_save_frames=False, debug_print_actions=False, debug_live_view=False, bomb_cooldown_steps=DEFAULT_BOMB_COOLDOWN_STEPS): # Added bomb_cooldown_steps
        print("[GeomEnv.__init__] ENTERED") # New debug print
        print(f"[GeomEnv.__init__] Received debug_live_view: {debug_live_view}") # New debug print
        print(f"[GeomEnv.__init__] Bomb cooldown steps: {bomb_cooldown_steps}")
        super(GeomEnv, self).__init__()
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.reward = 0.0
        self.step_reward = 0.0
        self.prev_reward = 0.0
        self.state = []
        
        # Debugging flags and parameters
        self.debug_mode = debug_mode # General debug flag, can be used by other logic if needed
        self.debug_save_frames = debug_save_frames
        self.debug_print_actions = debug_print_actions
        self.debug_live_view = debug_live_view
        self.last_action_components_for_debug = None
        self.debug_frame_save_freq = 0.001 # Approx 1 in 1000 frames
        self.debug_action_print_freq = 0.01 # Approx 1 in 100 actions
        self.debug_frames_dir = "debug_frames"

        if self.debug_save_frames and not os.path.exists(self.debug_frames_dir):
            os.makedirs(self.debug_frames_dir)
            print(f"Created directory for debug frames: {self.debug_frames_dir}")

        # Initialize window and screen capture
        self.window_handle = get_window_handle('Geometry Wars: Retro Evolved')
        if self.window_handle == 0:
            raise RuntimeError("Could not find Geometry Wars window")
        
        self.sct = mss.mss()
        self.game_window_lost = False # Initialize flag
        self._update_monitor_region() # Initial call to set up monitor region
        if self.game_window_lost: # Check if _update_monitor_region failed immediately
            raise RuntimeError("Failed to initialize monitor region in __init__, game window likely lost.")
        
        # Track game state
        self.prev_lives = 3  # Initial lives
        self.prev_score = 0
        self.steps_since_last_time_reward = 0
        self.frame_count = 0
        self.all_time_high_score_session = 0 # New: Tracks highest score in this session
        self.bomb_cooldown_steps = bomb_cooldown_steps # Store the cooldown
        self.bomb_action_overridden_this_step = False # New flag for callback

        # Define original action component dimensions
        self.action_component_dims = [3, 3, 3, 3, 2] 
        # Calculate the total number of discrete actions
        total_discrete_actions = np.prod(self.action_component_dims)
        self.action_space = spaces.Discrete(total_discrete_actions)

        # Update observation space shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATEY, STATEX, 1), dtype=np.uint8)
        print(f"[GeomEnv.__init__] Observation space set to: {self.observation_space.shape}") # Add print for confirmation

        self.y_move = [stop, up, down] # stop is the noop for y-axis movement
        self.x_move = [stop, left, right] # stop is also the noop for x-axis movement (it stops all movement)
                                         # This is okay, if agent wants to move only on y, x_move[0] will be stop().
                                         # If agent wants NO movement, both will be stop().
        
        self.y_shoot = [shoot_y_stop, shoot_up, shoot_down]
        self.x_shoot = [shoot_x_stop, shoot_left, shoot_right]
        
        self.bomb_actions = [bomb_specific_noop, bomb] # Use new bomb_specific_noop

        print(f"[GeomEnv.__init__] Initial prev_lives={self.prev_lives}, prev_score={self.prev_score}, all_time_high_score_session={self.all_time_high_score_session}") # Debug initial state

    def _update_monitor_region(self):
        """Update the monitor region to capture the entire detected gameplay content area."""
        if self.game_window_lost: return # Do nothing if window already known to be lost
        try:
            geom_window = gw.getWindowsWithTitle('Geometry Wars: Retro Evolved')[0]
            if geom_window is None:
                raise RuntimeError("Geometry Wars window not found by pygetwindow in _update_monitor_region.")
            horizontal_border_padding = 24 
            vertical_border_padding_top = 32  
            content_area_width = geom_window.width - horizontal_border_padding 
            content_area_height = geom_window.height - vertical_border_padding_top
            if content_area_width <= 0 or content_area_height <= 0:
                raise RuntimeError(f"Calculated content area non-positive. Win: w={geom_window.width}, h={geom_window.height}")
            content_area_screen_left = geom_window.left + (horizontal_border_padding / 2) 
            content_area_screen_top = geom_window.top + vertical_border_padding_top
            self.monitor = {
                'top': int(content_area_screen_top),
                'left': int(content_area_screen_left),
                'width': int(content_area_width),    
                'height': int(content_area_height)  
            }
            # print(f"[GeomEnv._update_monitor_region] Win(t={geom_window.top}, l={geom_window.left}, w={geom_window.width}, h={geom_window.height}), Content(w={content_area_width:.0f}, h={content_area_height:.0f}), Monitor (capturing full content): {self.monitor}") # SPAMMY - Commented out
        except (IndexError, AttributeError, RuntimeError) as e:
            if not self.game_window_lost: # Print only once if window becomes lost here
                print(f"ERROR in _update_monitor_region (game window now considered lost): {e}")
            self.game_window_lost = True

    def _get_observation(self):
        if self.game_window_lost:
            # print("[GeomEnv._get_observation] Game window previously lost...") # Can be noisy
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        try:
            self._update_monitor_region() 
            if self.game_window_lost: 
                 raise RuntimeError("Game window lost during _update_monitor_region called from _get_observation.")
            img_raw_full_content = self.sct.grab(self.monitor)
            img_np_full_content = np.asarray(img_raw_full_content)
            if img_np_full_content is None or img_np_full_content.size == 0:
                raise RuntimeError("Failed to capture screen (full content attempt)")
            img_resized = cv2.resize(img_np_full_content, (STATEX, STATEY), interpolation=cv2.INTER_AREA)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2GRAY)
            if self.debug_save_frames and np.random.rand() < self.debug_frame_save_freq:
                cv2.imwrite(os.path.join(self.debug_frames_dir, f"frame_resized_{time.time_ns()}.png"), img_resized)
            if self.debug_live_view:
                display_img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                # Reading lives/score here for display can be slow/error-prone if game state is iffy
                # Consider making this more robust or less frequent if it causes issues.
                try:
                    current_lives_display = getLives()
                    current_score_display = getScore()
                    state_text = f"L:{current_lives_display} S:{current_score_display}"
                    cv2.putText(display_img_bgr, state_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                except Exception as e_read:
                    cv2.putText(display_img_bgr, f"L/S Read Err: {e_read}", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,200),1)

                action_text_for_display = ""
                if self.last_action_components_for_debug is not None:
                    action_text_for_display = f"A:({self.last_action_components_for_debug[0]},{self.last_action_components_for_debug[1]},{self.last_action_components_for_debug[2]},{self.last_action_components_for_debug[3]},{self.last_action_components_for_debug[4]})"
                    cv2.putText(display_img_bgr, action_text_for_display, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imshow("Agent View (128x128 Downscaled)", display_img_bgr)
                cv2.waitKey(1)
            return np.expand_dims(img_gray, axis=-1) # Shape (STATEY, STATEX, 1)
        except Exception as e:
            if not self.game_window_lost: # Print only once
                 print(f"ERROR in _get_observation (game window now considered lost): {e}")
            self.game_window_lost = True 
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

    def _calculate_reward(self):
        if self.game_window_lost: 
            return 0.0 
        current_lives = getLives()
        current_score = getScore()
        reward = 0.0
        reward += SURVIVAL_REWARD
        if current_lives < self.prev_lives:
            reward += LIFE_LOST_PENALTY
        self.prev_lives = current_lives
        score_diff = current_score - self.prev_score
        if score_diff != 0:
            reward += score_diff * SCORE_REWARD_SCALE
        
        # New: Reward for new session high score
        if current_score > self.all_time_high_score_session:
            high_score_bonus = NEW_HIGH_SCORE_REWARD
            reward += high_score_bonus
            print(f"[GeomEnv._calculate_reward] NEW SESSION HIGH SCORE: {current_score} (Old: {self.all_time_high_score_session}). Added {high_score_bonus:.2f} bonus reward!")
            self.all_time_high_score_session = current_score
            
        self.prev_score = current_score
        self.steps_since_last_time_reward += 1
        if self.steps_since_last_time_reward >= TIME_REWARD_INTERVAL:
            reward += TIME_REWARD
            self.steps_since_last_time_reward = 0
        return reward

    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation and info."""
        super().reset(seed=seed)
        print("[GeomEnv.reset] Attempting to reset and start new game by pressing 'Enter' 5 times.")
        
        self.reward = 0.0
        self.step_reward = 0.0
        self.prev_reward = 0.0
        self.episode_start_time = time.time() # Add episode start time
        
        initial_game_window_lost_state = self.game_window_lost # Store state before trying to re-acquire
        self.game_window_lost = False # Optimistically try to reset flag

        self.window_handle = get_window_handle('Geometry Wars: Retro Evolved')
        if self.window_handle == 0:
            if not initial_game_window_lost_state: # Print only if it just became lost
                 print("[GeomEnv.reset] Game window not found at reset. Episode cannot start.")
            self.game_window_lost = True
            dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return dummy_obs, {"error": "Game window not found at reset"}

        num_enter_presses = 3
        delay_between_presses = 0.1 # seconds, adjust if needed
        delay_after_sequence = 1.0 # seconds, for game to fully load/start

        try:
            for i in range(num_enter_presses):
                print(f"[GeomEnv.reset] Sending 'Enter' press {i+1}/{num_enter_presses}...")
                enter(self.window_handle)
                time.sleep(delay_between_presses)
            time.sleep(delay_after_sequence)
        except InvalidWindowHandleError as e:
            if not initial_game_window_lost_state: # Print only if it just became lost
                print(f"[GeomEnv.reset] Invalid window handle during reset key presses: {e}. Episode cannot start.")
            self.game_window_lost = True
            dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return dummy_obs, {"error": "Invalid window handle during reset key presses"}
        
        # If window was lost before, but we successfully sent keys, it might be back
        # if initial_game_window_lost_state and not self.game_window_lost:
        # print("[GeomEnv.reset] Game window potentially reacquired after reset sequence.")

        self.prev_lives = getLives() 
        self.prev_score = getScore()
        print(f"[GeomEnv.reset] After Enter sequence, initial read: prev_lives={self.prev_lives}, prev_score={self.prev_score}, current session high: {self.all_time_high_score_session}")
        
        if not (self.prev_lives == 3 and self.prev_score < 500): # Allow some minimal score if game auto-gives some points
            print(f"WARNING [GeomEnv.reset]: Game may not be in a clean starting state. Lives: {self.prev_lives}, Score: {self.prev_score}")

        self.steps_since_last_time_reward = 0
        self.frame_count = 0 # Ensure frame_count is reset
        self.last_action_components_for_debug = None
        self.state = self._get_observation() 
        if self.game_window_lost: 
            # _get_observation already printed its error if it just became lost
            # print("[GeomEnv.reset] Failed to get initial observation after reset.")
            return self.state, {"error": "Failed initial observation after reset"} 

        # print(f"[GeomEnv.reset] Reset complete. Initial state shape: {self.state.shape}")
        return self.state, {}

    def step(self, action):
        """Execute one time step within the environment."""
        if self.game_window_lost:
            # print("[GeomEnv.step] Game window lost. Returning as terminated.") # Can be noisy
            dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return dummy_obs, 0.0, True, False, {"error": "Game window lost before step execution"}

        self.frame_count += 1
        action_components_tuple = np.unravel_index(action, self.action_component_dims)
        action_components = [comp.item() for comp in action_components_tuple] 

        if self.debug_print_actions and np.random.rand() < self.debug_action_print_freq:
            print(f"Step: {self.frame_count}, Raw: {action}, Decoded: {action_components}")
        self.last_action_components_for_debug = action_components 
        self.bomb_action_overridden_this_step = False # Reset at the start of action processing
            
        if self.frame_count % FRAME_SKIP != 0:
            return self.state, 0, False, False, {}
            
        try:
            # Execute movement and shooting actions first
            self.y_move[action_components[0]](self.window_handle) 
            self.x_move[action_components[1]](self.window_handle)
            self.y_shoot[action_components[2]](self.window_handle)
            self.x_shoot[action_components[3]](self.window_handle)

            # Handle bomb action, considering cooldown
            chosen_bomb_action_idx = action_components[4] # 0 for bomb_specific_noop, 1 for bomb

            if chosen_bomb_action_idx == 1 and self.frame_count <= self.bomb_cooldown_steps:
                # Agent wants to bomb (idx 1), but is on cooldown.
                self.bomb_action_overridden_this_step = True 
                if self.debug_print_actions and np.random.rand() < 0.1: 
                     print(f"[GeomEnv.step] Bomb usage attempted at episode step {self.frame_count} but denied (cooldown active: {self.bomb_cooldown_steps} steps). Bomb part of action is a true no-op.")
                # No explicit call here, meaning the bomb is not activated. Movement/shooting already done.
            else:
                # Agent either chose bomb_specific_noop (idx 0), 
                # OR wants to bomb (idx 1) and is NOT on cooldown.
                # Execute the chosen bomb action (which will be bomb_specific_noop() or bomb()).
                self.bomb_actions[chosen_bomb_action_idx](self.window_handle)

        except InvalidWindowHandleError as e:
            if not self.game_window_lost: 
                print(f"[GeomEnv.step] Invalid window handle during action execution: {e}. Terminating.")
            self.game_window_lost = True
            dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return dummy_obs, 0.0, True, False, {"error": "Invalid window handle during action execution"}
        
        self.state = self._get_observation()
        if self.game_window_lost: 
            # print("[GeomEnv.step] Failed to get observation after actions. Terminating.") # Can be noisy
            return self.state, 0.0, True, False, {"error": "Failed observation after action"} 

        reward = self._calculate_reward()
        terminated = self.prev_lives <= 0 or self.game_window_lost
        truncated = False

        info = {} # Initialize info dictionary
        if terminated:
            current_time = time.time()
            info['episode'] = {
                'r': reward, # Using the final step's shaped reward for consistency with Monitor's 'r'
                'l': self.frame_count,
                't': current_time - self.episode_start_time,
                'final_game_score': self.prev_score, # Actual game score at end of episode
                'session_high_game_score': self.all_time_high_score_session # Session high game score
            }

        return self.state, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the current state."""
        if self.game_window_lost and self.viewer is not None: 
            self.viewer.close()
            self.viewer = None
            return False
        if mode == 'human':
            if self.viewer is None and not self.game_window_lost:
                from gymnasium.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            if self.viewer is not None:
                img_display = self.state if self.state.shape[-1] == 3 else np.repeat(self.state, 3, axis=-1)
                self.viewer.imshow(img_display)
                return self.viewer.isopen
        return None

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.sct is not None:
            self.sct.close()
        if self.debug_live_view: 
            print("[GeomEnv.close] Attempting to destroy 'Agent View' window.")
            try:
                cv2.destroyWindow("Agent View (128x128 Downscaled)") # Be specific
                cv2.waitKey(1) # Allow time for destroy
            except cv2.error as e:
                print(f"[GeomEnv.close] cv2.destroyWindow error (ignorable if window wasn't shown): {e}")
                pass
        print("Environment closed.")





