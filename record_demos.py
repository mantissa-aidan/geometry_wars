import gymnasium
import geom_wars_env # Ensure environment is registered
import numpy as np
import time
import os
import pickle
import keyboard # For capturing keyboard input. May require: pip install keyboard

# --- Configuration ---
OUTPUT_DEMO_FILE = "human_demos.pkl"
EPISODES_TO_RECORD = 5 # Number of episodes to record in one go
GAME_START_COUNTDOWN = 5 # Seconds to get ready and focus the game window

# Define key mappings (adjust if your preferred keys are different)
# Movement
KEY_MOVE_UP = 'w'
KEY_MOVE_DOWN = 's'
KEY_MOVE_LEFT = 'a'
KEY_MOVE_RIGHT = 'd'

# Shooting
KEY_SHOOT_UP = 'i' # or 'up' for arrow key
KEY_SHOOT_DOWN = 'k' # or 'down'
KEY_SHOOT_LEFT = 'j' # or 'left'
KEY_SHOOT_RIGHT = 'l' # or 'right'

# Bomb
KEY_BOMB = 'space'

# Quit recording
KEY_QUIT_RECORDING_SESSION = 'escape'


def get_human_action_and_discrete(action_component_dims):
    """
    Captures human keyboard input and translates it into the 5-component action
    and the corresponding single discrete action integer.

    Returns:
        tuple: (action_components, discrete_action_idx)
               action_components is a list like [y_move, x_move, y_shoot, x_shoot, bomb_action]
               discrete_action_idx is the integer action.
               Returns (None, None) if no relevant keys are pressed for a meaningful game action
               (e.g., only movement without shooting might be considered 'noop' for some components).
               For simplicity, we map any combination to an action.
    """
    y_move_idx = 0  # 0: noop, 1: up, 2: down
    if keyboard.is_pressed(KEY_MOVE_UP):
        y_move_idx = 1
    elif keyboard.is_pressed(KEY_MOVE_DOWN):
        y_move_idx = 2

    x_move_idx = 0  # 0: noop, 1: left, 2: right
    if keyboard.is_pressed(KEY_MOVE_LEFT):
        x_move_idx = 1
    elif keyboard.is_pressed(KEY_MOVE_RIGHT):
        x_move_idx = 2

    y_shoot_idx = 0  # 0: noop, 1: up, 2: down
    if keyboard.is_pressed(KEY_SHOOT_UP):
        y_shoot_idx = 1
    elif keyboard.is_pressed(KEY_SHOOT_DOWN):
        y_shoot_idx = 2
    
    x_shoot_idx = 0  # 0: noop, 1: left, 2: right
    if keyboard.is_pressed(KEY_SHOOT_LEFT):
        x_shoot_idx = 1
    elif keyboard.is_pressed(KEY_SHOOT_RIGHT):
        x_shoot_idx = 2

    bomb_idx = 0  # 0: noop, 1: bomb
    if keyboard.is_pressed(KEY_BOMB):
        bomb_idx = 1

    action_components = [y_move_idx, x_move_idx, y_shoot_idx, x_shoot_idx, bomb_idx]
    
    # Convert components to single discrete action index
    # This uses the same logic as np.ravel_multi_index, assuming action_component_dims is [dim_0, dim_1, ..., dim_n-1]
    # discrete_idx = sum(action_components[i] * prod(action_component_dims[i+1:]) for i in range(len_dims))
    # Or more directly using np.ravel_multi_index if available and easy.
    # For SB3, the action is just the integer. We need np.ravel_multi_index.
    discrete_action_idx = np.ravel_multi_index(tuple(action_components), action_component_dims)
    
    return action_components, int(discrete_action_idx)

def record_demonstrations():
    """
    Main function to record human gameplay demonstrations.
    """
    print("Starting demonstration recording session.")
    print(f"Demos will be saved to: {OUTPUT_DEMO_FILE}")
    print(f"Press '{KEY_QUIT_RECORDING_SESSION}' during gameplay (between steps) to stop recording early.")
    print(f"IMPORTANT: The 'keyboard' library might require running this script with administrator/sudo privileges.")
    print("Ensure the Geometry Wars game window is active and focused after the countdown.")

    env = None
    try:
        # Initialize environment - enable live view so the player can see
        env = gymnasium.make('GeomWarsEnv-v0', debug_live_view=True)
        print("Environment created.")
        
        # Get action_component_dims from the environment instance
        # Ensure this attribute exists and is correctly set in your GeomEnv
        if not hasattr(env.unwrapped, 'action_component_dims'):
            print("ERROR: env.unwrapped.action_component_dims not found! Cannot map actions.")
            return
        action_dims = env.unwrapped.action_component_dims
        print(f"Action component dimensions from env: {action_dims}")


        # Countdown
        for i in range(GAME_START_COUNTDOWN, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("Recording NOW! Focus the game window.")

        all_transitions = []
        for episode_num in range(EPISODES_TO_RECORD):
            print(f"--- Starting Episode {episode_num + 1} / {EPISODES_TO_RECORD} ---")
            obs, info = env.reset()
            
            terminated = False
            truncated = False
            total_episode_reward = 0
            episode_steps = 0

            while not (terminated or truncated):
                if keyboard.is_pressed(KEY_QUIT_RECORDING_SESSION):
                    print("Quit key pressed. Ending recording session.")
                    # Manually trigger termination for saving logic
                    terminated = True 
                    break 

                # Get human action
                action_components, discrete_action = get_human_action_and_discrete(action_dims)
                
                # The environment's step function will handle sending input to the game
                next_obs, reward, terminated, truncated, info = env.step(discrete_action)
                
                # Store the transition for SB3 replay buffer (obs, next_obs, action, reward, done, infos)
                # 'done' for SB3 is 'terminated' (True if episode ends due to termination condition)
                # For SB3, 'infos' is usually a list of dicts, often empty if not used for specific logic.
                all_transitions.append({
                    'obs': obs,
                    'next_obs': next_obs,
                    'action': np.array(discrete_action), # Ensure action is a NumPy scalar or array
                    'reward': np.array(reward),         # Ensure reward is a NumPy scalar or array
                    'done': np.array(terminated),         # Use terminated for 'done'
                    'infos': [{}] # SB3 expects a list of info dicts, often just [{}]
                })

                obs = next_obs
                total_episode_reward += reward
                episode_steps += 1

                # Brief pause to allow keyboard library to register key releases
                # and to make recording somewhat paced like human reaction time.
                time.sleep(0.05) # Adjust as needed, e.g., 1/20th of a second

            print(f"Episode {episode_num + 1} finished. Steps: {episode_steps}, Reward: {total_episode_reward:.2f}")
            if keyboard.is_pressed(KEY_QUIT_RECORDING_SESSION): # Check again if quit during episode end
                break
            
            if not (terminated or truncated): # Should not happen if loop broke due to quit key
                 print("Warning: Episode loop exited without terminated or truncated or quit signal.")


        # Save demonstrations
        if all_transitions:
            if os.path.exists(OUTPUT_DEMO_FILE):
                print(f"Appending to existing demo file: {OUTPUT_DEMO_FILE}")
                with open(OUTPUT_DEMO_FILE, 'rb') as f:
                    existing_transitions = pickle.load(f)
                all_transitions = existing_transitions + all_transitions
            
            with open(OUTPUT_DEMO_FILE, 'wb') as f:
                pickle.dump(all_transitions, f)
            print(f"Successfully saved/appended {len(all_transitions)} transitions to {OUTPUT_DEMO_FILE}")
        else:
            print("No transitions recorded.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env:
            env.close()
            print("Environment closed.")
        print("Recording script finished.")

if __name__ == "__main__":
    record_demonstrations() 