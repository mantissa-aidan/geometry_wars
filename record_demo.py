import numpy as np
import time
import keyboard
import pickle
import os
from datetime import datetime
from gym.envs.aidan_envs.geom_wars import GeomEnv
from gym.utils.window_input import (
    get_window_handle, up, down, left, right, stop,
    shoot_up, shoot_down, shoot_left, shoot_right, shoot_stop,
    bomb, enter
)
from gym.utils.read_memory import getScore

# Constants
DEMOS_DIR = "demos"
MAX_DEMOS_PER_FILE = 10  # Maximum number of demonstrations to store in one file

class DemoRecorder:
    def __init__(self):
        self.env = GeomEnv()
        self.demos = []
        self.current_demo = []
        self.recording = False
        self.demo_count = 0
        
        # Create demos directory if it doesn't exist
        if not os.path.exists(DEMOS_DIR):
            os.makedirs(DEMOS_DIR)
        
        # Action mapping for keyboard input
        self.action_map = {
            'w': 1,  # up
            's': 2,  # down
            'a': 1,  # left
            'd': 2,  # right
            'up': 1,    # shoot up
            'down': 2,  # shoot down
            'left': 1,  # shoot left
            'right': 2, # shoot right
            'space': 1  # bomb
        }
        
        # Default action (no movement)
        self.current_action = [0, 0, 0, 0, 0]
        
    def _get_action_from_keys(self):
        """Convert keyboard state to action array."""
        action = [0, 0, 0, 0, 0]  # [y_move, x_move, y_shoot, x_shoot, bomb]
        
        # Movement
        if keyboard.is_pressed('w'): action[0] = 1  # up
        elif keyboard.is_pressed('s'): action[0] = 2  # down
        
        if keyboard.is_pressed('a'): action[1] = 1  # left
        elif keyboard.is_pressed('d'): action[1] = 2  # right
        
        # Shooting
        if keyboard.is_pressed('up'): action[2] = 1  # shoot up
        elif keyboard.is_pressed('down'): action[2] = 2  # shoot down
        
        if keyboard.is_pressed('left'): action[3] = 1  # shoot left
        elif keyboard.is_pressed('right'): action[3] = 2  # shoot right
        
        # Bomb
        if keyboard.is_pressed('space'): action[4] = 1
        
        return action
    
    def start_recording(self):
        """Start recording a new demonstration."""
        print("\nStarting new demonstration...")
        print("Controls:")
        print("  Movement: WASD")
        print("  Shooting: Arrow keys")
        print("  Bomb: Space")
        print("  Press 'r' to stop recording")
        print("  Press ENTER when you are at the start of a new game (score = 0)")
        input()  # Wait for user to press ENTER
        # Ready check: ensure score is 0
        score = getScore()
        if score != 0:
            print(f"[WARN] Score is {score}, not 0. Please restart the game and try again.")
            return
        
        self.recording = True
        self.current_demo = []
        state = self.env.reset()
        
        while self.recording:
            # Get action from keyboard
            action = self._get_action_from_keys()
            
            # Step environment
            next_state, reward, done, _ = self.env.step(action)
            
            # Record transition
            self.current_demo.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            state = next_state
            
            # Check for recording stop
            if keyboard.is_pressed('r'):
                self.recording = False
            
            # Check for game over
            if done:
                print("Game over! Starting new episode...")
                state = self.env.reset()
            
            time.sleep(0.016)  # ~60 FPS
        
        # Save demonstration
        if self.current_demo:
            self.demos.append(self.current_demo)
            self.demo_count += 1
            print(f"Saved demonstration {self.demo_count} with {len(self.current_demo)} steps")
            
            # Auto-save if we've reached the maximum demos per file
            if len(self.demos) >= MAX_DEMOS_PER_FILE:
                self.save_demos()
    
    def save_demos(self, filename=None):
        """Save all recorded demonstrations to a file."""
        if not self.demos:
            print("No demonstrations to save!")
            return
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"demos_{timestamp}.pkl"
        
        # Ensure filename is in the demos directory
        filepath = os.path.join(DEMOS_DIR, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.demos, f)
        print(f"Saved {len(self.demos)} demonstrations to {filepath}")
        
        # Clear the demos list after saving
        self.demos = []

def main():
    recorder = DemoRecorder()
    
    print("Geometry Wars Demonstration Recorder")
    print("Controls:")
    print("  'q' - Start recording")
    print("  'r' - Stop recording")
    print("  's' - Save current demonstrations")
    print("  'x' - Exit")
    print("\nDemonstrations will be saved in the 'demos' directory")
    print("Auto-saving will occur after every", MAX_DEMOS_PER_FILE, "demonstrations")
    
    while True:
        if keyboard.is_pressed('q'):
            recorder.start_recording()
        elif keyboard.is_pressed('s'):
            recorder.save_demos()
        elif keyboard.is_pressed('x'):
            # Save any remaining demos before exiting
            if recorder.demos:
                recorder.save_demos()
            break
        time.sleep(0.1)

if __name__ == "__main__":
    main() 