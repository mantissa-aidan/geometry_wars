import gymnasium
import geom_wars_env # Import your package to ensure registration. This line executes geom_wars_env/__init__.py which in turn imports geom_wars_env/envs/__init__.py where registration happens.
import numpy as np # For np.unravel_index in callback
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback # Import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer # For type hinting if needed, and understanding structure
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack # Added imports
from collections import deque # Added deque
import time
import os
import argparse # Import argparse
import pickle # For loading demonstrations
import torch # Added torch import

# Define a consistent path for the model to load from and save to for resuming
RESUME_MODEL_FILENAME = "geom_wars_dqn_resume.zip"
DEFAULT_DEMO_FILE = "human_demos.pkl"
DEFAULT_ACTION_LOG_INTERVAL = 20 # Default interval for the new custom logger - Changed from 250

# --- Custom Callback for logging actions and rewards ---
class ActionRewardLoggerCallback(BaseCallback):
    """
    A custom callback that logs the agent's actions and step rewards.
    """
    def __init__(self, print_interval=DEFAULT_ACTION_LOG_INTERVAL, verbose=0):
        super(ActionRewardLoggerCallback, self).__init__(verbose)
        self.print_interval = print_interval
        self._action_component_dims = None
        # No need to store bomb_action_overridden_this_step here, get it fresh from env

    def _on_training_start(self) -> None:
        # Attempt to get action_component_dims from the environment
        try:
            # Access the underlying environment, potentially skipping wrappers like Monitor
            if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0: # For VecEnvs
                actual_env = self.training_env.envs[0].unwrapped
            else: # For single environment
                actual_env = self.training_env.unwrapped
            
            if hasattr(actual_env, 'action_component_dims'):
                self._action_component_dims = actual_env.action_component_dims
                print(f"[ActionRewardLoggerCallback] Successfully fetched action_component_dims: {self._action_component_dims}")
            else:
                print("[ActionRewardLoggerCallback] Warning: env.unwrapped.action_component_dims not found. Will not print decoded actions.")
        except Exception as e:
            print(f"[ActionRewardLoggerCallback] Error accessing action_component_dims: {e}. Will not print decoded actions.")


    def _on_step(self) -> bool:
        if self.print_interval > 0 and self.n_calls % self.print_interval == 0:
            action = self.locals['actions'][0] 
            reward = self.locals['rewards'][0] 
            
            decoded_action_str = ""
            bomb_status_str = "" # For indicating bomb override

            if self._action_component_dims is not None:
                try:
                    decoded_components_list = list(np.unravel_index(action, self._action_component_dims))
                    decoded_action_str = f" Decoded: {decoded_components_list}"

                    # Check for bomb override
                    # Access the underlying environment, potentially skipping wrappers like Monitor
                    if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0: # For VecEnvs (should be 1 for us)
                        actual_env = self.training_env.envs[0].unwrapped
                    else: # For single environment
                        actual_env = self.training_env.unwrapped

                    if hasattr(actual_env, 'bomb_action_overridden_this_step'):
                        if decoded_components_list[4] == 1 and actual_env.bomb_action_overridden_this_step:
                            bomb_status_str = " (Bomb DENIED by Cooldown)"
                    else:
                        # This case should ideally not happen if the env is correctly set up
                        print("[ActionRewardLoggerCallback] Warning: env.unwrapped.bomb_action_overridden_this_step not found.")

                except ValueError: 
                    decoded_action_str = " (decode error)"

            print(f"  [Log] Step: {self.num_timesteps:<8} Action: {action:<3}{decoded_action_str}{bomb_status_str} Reward: {reward:.2f}")
        return True


def train_sb3(env_id='GeomWarsEnv-v0', 
              total_timesteps=10000000, # Default to a large number for "indefinite"
              log_dir="./sb3_geom_logs/",
              load_demos_file=None, # New parameter for demo file path
              enable_debug_save_frames=False, # New parameter
              enable_debug_print_actions=False, # New parameter
              enable_debug_live_view=False, # New parameter for live view
              countdown_seconds=5, # New parameter for countdown
              device="auto", # New parameter for device selection
              action_log_interval=DEFAULT_ACTION_LOG_INTERVAL, # New parameter for callback
              n_stack_frames=2, # Changed default from 3 to 2
              continue_training_new_phase=False # New parameter for this feature
              ):
    """
    Train a DQN agent using Stable Baselines3, with resume and demo loading capability.
    """
    print(f"Debug Save Frames: {enable_debug_save_frames}, Debug Print Actions: {enable_debug_print_actions}, Debug Live View: {enable_debug_live_view}")
    print(f"Attempting to load demonstrations from: {load_demos_file if load_demos_file else 'Not specified'}")
    print(f"Action/Reward Log Interval: {action_log_interval if action_log_interval > 0 else 'Disabled'}")
    print(f"Frame Stacking: {n_stack_frames} frames") # Log frame stacking
    print(f"Requested training device: {device}")
    print(f"Continue training new phase from existing weights: {continue_training_new_phase}") # Log new flag
    
    # --- Graceful CUDA failure check ---
    effective_device = device
    if device == "cuda":
        if not torch.cuda.is_available():
            print("""
            ********************************************************************************
            ERROR: CUDA device requested (device='cuda') but CUDA is not available!
            Please ensure:
            1. You have an NVIDIA GPU.
            2. NVIDIA drivers are installed and up to date.
            3. CUDA Toolkit is installed (compatible version with PyTorch).
            4. PyTorch was installed with CUDA support (not the CPU-only version).
               You can check with: python -c "import torch; print(torch.cuda.is_available())"
            Falling back to CPU. If you want to enforce GPU or fail, please adjust script.
            ********************************************************************************
            """)
            effective_device = "cpu" # Fallback to CPU with a warning
            # Alternatively, to strictly fail: 
            # print("Exiting due to CUDA unavailability.")
            # return 
        else:
            print("CUDA is available. Proceeding with CUDA device.")
    elif device == "auto":
        if torch.cuda.is_available():
            print("Device 'auto' selected and CUDA is available. Will attempt to use CUDA.")
            effective_device = "cuda"
        else:
            print("Device 'auto' selected but CUDA is not available. Will use CPU.")
            effective_device = "cpu"
    else: # CPU explicitly requested
        print("Proceeding with CPU device.")
        effective_device = "cpu"
    
    # Ensure log directory exists (it should from main)
    os.makedirs(log_dir, exist_ok=True)
    # Corrected model_resume_path logic: It should use a fixed base directory, not the session log_dir.
    # This requires main_model_storage_dir to be known here, or passed differently.
    # For now, let's assume log_dir IS the session log dir, and main_model_storage_dir is its parent.
    main_model_storage_dir = os.path.dirname(os.path.dirname(log_dir)) # e.g., ./sb3_geom_logs/DQN_ResumableModel
    if not os.path.basename(main_model_storage_dir) == "DQN_ResumableModel": # Heuristic check
        # This might happen if log_dir structure changes. Fallback or warning needed.
        print(f"Warning: main_model_storage_dir derived as {main_model_storage_dir}, may not be correct.")

    model_resume_path = os.path.join(main_model_storage_dir, RESUME_MODEL_FILENAME)

    # Create the environment
    try:
        # Create the base environment
        base_env = gymnasium.make(
            env_id,
            debug_save_frames=enable_debug_save_frames,
            debug_print_actions=enable_debug_print_actions,
            debug_live_view=enable_debug_live_view 
        )
        print(f"Base env observation space: {base_env.observation_space.shape}")

        # 1. Wrap base_env with Monitor FIRST
        monitored_base_env = Monitor(base_env, log_dir) 

        # 2. Wrap monitored_base_env with DummyVecEnv
        vec_env = DummyVecEnv([lambda: monitored_base_env])
        
        # 3. Wrap vec_env with VecFrameStack
        env = VecFrameStack(vec_env, n_stack=n_stack_frames)
        print(f"Stacked env observation space: {env.observation_space.shape}")

    except gymnasium.error.NameNotFound:
        print(f"Error: Environment ID '{env_id}' not found. Make sure it's registered.")
        print("Ensure that 'geom_wars_env' is importable and its __init__.py files are correctly set up.")
        # For debugging, print available environments:
        # from gymnasium.envs.registration import registry
        # print("Available environments:", list(registry.keys()))
        return

    # The `env` variable that the model uses is now the fully wrapped VecFrameStack environment.

    model = None
    reset_timesteps_for_learn = False # Default to False

    if continue_training_new_phase:
        if os.path.exists(model_resume_path):
            print(f"Starting NEW TRAINING PHASE. Loading weights from {model_resume_path}...")
            try:
                model = DQN.load(model_resume_path, env=env, tensorboard_log=log_dir, device=effective_device)
                print(f"Model weights loaded. Original total timesteps: {model.num_timesteps}. Schedules will be reset for this phase.")
                reset_timesteps_for_learn = True
            except Exception as e:
                print(f"Error loading model for new phase: {e}. Creating a new model instead.")
                model = None # Ensure model is None if loading failed
        else:
            print(f"Error: --new_phase_from_existing_weights specified, but model {model_resume_path} not found! Creating a new model from scratch.")
            model = None # Fall through to new model creation
    
    elif os.path.exists(model_resume_path): # Standard resume (not a new phase)
        print(f"Resuming standard training. Loading model from {model_resume_path}...")
        try:
            model = DQN.load(model_resume_path, env=env, tensorboard_log=log_dir, device=effective_device)
            print(f"Model loaded. It has been trained for {model.num_timesteps} timesteps already.")
            # reset_timesteps_for_learn remains False for standard resume
        except Exception as e:
            print(f"Error loading model for standard resume: {e}. Creating a new model.")
            model = None
    
    if model is None: # Create new model if no resume path, or if new_phase load failed/was specified but no model
        print("No existing model found or loading failed. Creating a new model...")
        model = DQN("CnnPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=log_dir, # This is the session_log_dir
                    learning_rate=0.00025,
                    gamma=0.99,
                    buffer_size=100000,
                    batch_size=64,
                    learning_starts=1000,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.01,
                    train_freq=4,
                    gradient_steps=1,
                    target_update_interval=10000,
                    device=effective_device
        )
        print(f"New model created. Using device: {model.device}")
        reset_timesteps_for_learn = False # It's a brand new model, schedules start from 0 naturally

        # Pre-fill replay buffer if demo file is provided and model is new
        if load_demos_file and os.path.exists(load_demos_file):
            print(f"Loading demonstrations from {load_demos_file} to pre-fill replay buffer...")
            try:
                with open(load_demos_file, 'rb') as f:
                    demonstrations = pickle.load(f)
                
                if not demonstrations:
                    print("Demonstration file is empty. Starting with an empty replay buffer.")
                else:
                    num_added_to_buffer = 0
                    if n_stack_frames > 1:
                        print(f"Processing demonstrations for frame stacking (n_stack={n_stack_frames})...")
                        sample_obs_from_demo = demonstrations[0]['obs'] # Assuming at least one demo if not empty
                        zero_frame = np.zeros_like(sample_obs_from_demo)
                        
                        current_frames_deque = deque(maxlen=n_stack_frames)
                        for _ in range(n_stack_frames):
                            current_frames_deque.append(zero_frame)

                        for demo_transition in demonstrations:
                            s_single = demo_transition['obs']
                            action = demo_transition['action']
                            reward = demo_transition['reward']
                            s_prime_single = demo_transition['next_obs']
                            done = demo_transition['done']
                            infos_dict = demo_transition['infos']

                            current_frames_deque.append(s_single)
                            obs_stacked_arr = np.concatenate(list(current_frames_deque), axis=-1)

                            next_frames_for_stack_temp_deque = deque(maxlen=n_stack_frames)
                            if done:
                                for _ in range(n_stack_frames):
                                    next_frames_for_stack_temp_deque.append(s_prime_single)
                            else:
                                for frame_idx in range(1, n_stack_frames):
                                    next_frames_for_stack_temp_deque.append(current_frames_deque[frame_idx])
                                next_frames_for_stack_temp_deque.append(s_prime_single)
                            next_obs_stacked_arr = np.concatenate(list(next_frames_for_stack_temp_deque), axis=-1)
                            
                            action_arr = np.array(action) # DQN replay buffer expects actions as np.ndarray
                            infos_list = [infos_dict if infos_dict is not None else {}]

                            model.replay_buffer.add(obs_stacked_arr, next_obs_stacked_arr, action_arr, reward, done, infos_list)
                            num_added_to_buffer += 1

                            if done:
                                current_frames_deque.clear()
                                for _ in range(n_stack_frames):
                                    current_frames_deque.append(s_prime_single) # Reset for next episode start
                        
                        print(f"Successfully processed and added {num_added_to_buffer} transitions from demonstrations to replay buffer with frame stacking.")

                    else: # n_stack_frames <= 1 (no frame stacking or trivial)
                        for demo_transition in demonstrations:
                            obs = demo_transition['obs']
                            next_obs = demo_transition['next_obs']
                            action = np.array(demo_transition['action']) 
                            reward = demo_transition['reward'] 
                            done = demo_transition['done']
                            infos_dict = demo_transition['infos']
                            infos_list = [infos_dict if infos_dict is not None else {}]
                            model.replay_buffer.add(obs, next_obs, action, reward, done, infos_list)
                            num_added_to_buffer += 1
                        print(f"Successfully loaded and added {num_added_to_buffer} transitions (no/trivial frame stacking).")
                    
                    print(f"Replay buffer current size: {model.replay_buffer.size()}/{model.replay_buffer.buffer_size}") # Use .size() method

            except Exception as e:
                print(f"Error loading or processing demonstrations: {e}")
                import traceback
                traceback.print_exc()
        elif load_demos_file:
            print(f"Demonstration file {load_demos_file} not found. Starting with an empty replay buffer.")

    # Callbacks
    callbacks_list = []
    
    # Checkpoint Callback
    checkpoint_save_path = os.path.join(log_dir, "checkpoints") # Separate subdir for checkpoints
    os.makedirs(checkpoint_save_path, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=50000, # Save checkpoints less frequently if main save is frequent
                                             save_path=checkpoint_save_path,
                                             name_prefix='geom_wars_dqn_ckpt')
    callbacks_list.append(checkpoint_callback)

    # Custom Action/Reward Logger Callback
    if action_log_interval > 0:
        custom_logger_callback = ActionRewardLoggerCallback(print_interval=action_log_interval)
        callbacks_list.append(custom_logger_callback)

    print(f"TensorBoard logs for this session will be saved to: {log_dir}")
    print(f"Models for resuming (from any phase) will be saved to: {model_resume_path}")
    print(f"Periodic checkpoints for THIS session will be saved to: {checkpoint_save_path}")

    # Countdown before starting training
    if countdown_seconds > 0:
        print(f"\n!!! GET READY TO START THE GAME MANUALLY (if new game needed) !!!")
        print(f"Training will begin in {countdown_seconds} seconds.")
        for i in range(countdown_seconds, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("Starting training NOW!")

    try:
        # SB3's learn() method will continue from the model's current num_timesteps if loaded.
        # The total_timesteps here means "additional steps to train for this session".
        model.learn(total_timesteps=total_timesteps, callback=callbacks_list, log_interval=10, reset_num_timesteps=reset_timesteps_for_learn)
    except Exception as e:
        print(f"An error occurred during model.learn(): {e}")
        env.close()
        # Save the current model state even if an error occurs during learn, for potential resume
        print(f"Attempting to save model to {model_resume_path} after error...")
        model.save(model_resume_path)
        print("Model saved after error.")
        return

    print(f"Training session finished. Saving final model to {model_resume_path}...")
    model.save(model_resume_path)
    print(f"Model successfully saved to {model_resume_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for Geometry Wars.")
    parser.add_argument("--timesteps", type=int, default=10000000, help="Total number of training timesteps for this session (default 10 million for long run).")
    parser.add_argument("--log_suffix", type=str, default=None, help="Suffix for the log directory name.")
    parser.add_argument("--load_demos_file", type=str, default=None, help=f"Path to demonstration file to load (e.g., {DEFAULT_DEMO_FILE}). Default is None (disabled).")
    parser.add_argument("--save_frames", action='store_true', help="Enable saving debug frames from the environment.")
    parser.add_argument("--print_actions_env", action='store_true', help="Enable environment's detailed action printing (can be verbose).")
    parser.add_argument("--live_view", action='store_true', help="Enable live view of agent's observation and chosen action.") # New argument
    parser.add_argument("--env_id", type=str, default='GeomWarsEnv-v0', help="Gymnasium Environment ID.")
    parser.add_argument("--countdown", type=int, default=5, help="Countdown seconds.")
    parser.add_argument("--device", type=str, default="auto", choices=['auto', 'cuda', 'cpu'], help="Device to use for training (auto, cuda, cpu).")
    parser.add_argument("--action_log_interval", type=int, default=DEFAULT_ACTION_LOG_INTERVAL, help="Interval (steps) for logging action/reward. 0 to disable.")
    parser.add_argument("--n_stack_frames", type=int, default=3, help="Number of frames to stack for observations (default: 3). Set to 1 to disable frame stacking effectively (though VecFrameStack might still wrap).")
    parser.add_argument("--new_phase_from_existing_weights", action='store_true', 
                        help="Load weights from RESUME_MODEL_FILENAME, but reset schedules and log as a new training phase.")

    args = parser.parse_args()

    # Log directory setup (remains similar, but base name is fixed for resume)
    # The timestamped subdirectory will be for this specific run's TensorBoard logs & checkpoints
    # The resumable model will be at the root of sb3_geom_logs/
    base_log_dir = "./sb3_geom_logs/"
    specific_run_log_subdir_name = time.strftime("%Y%m%d-%H%M%S")
    if args.log_suffix:
        specific_run_log_subdir_name = f"{args.log_suffix}_{specific_run_log_subdir_name}"
    
    # The main log_dir for TensorBoard and where the resumable model lives
    # For simplicity, let's use a fixed name for the directory containing the resumable model
    # and TensorBoard can show multiple runs within subdirs if needed.
    # Let's have one main log_dir, and the resumable model lives there.
    # Tensorboard logs and checkpoints for specific runs can go into timestamped subdirs.

    # Revised log strategy:
    # - ./sb3_geom_logs/DQN_ResumableModel/ (this dir will contain geom_wars_dqn_resume.zip)
    # - ./sb3_geom_logs/DQN_ResumableModel/runs/YYYYMMDD-HHMMSS_suffix/ (for TB logs and checkpoints of specific sessions)

    main_model_dir = os.path.join(base_log_dir, "DQN_ResumableModel")
    os.makedirs(main_model_dir, exist_ok=True)

    # Tensorboard logs and checkpoints for THIS specific session
    session_log_dir = os.path.join(main_model_dir, "runs", specific_run_log_subdir_name)
    os.makedirs(session_log_dir, exist_ok=True)

    # Determine if demo loading should be attempted
    demos_to_load = args.load_demos_file
    if demos_to_load and demos_to_load.strip() == "": # Handle empty string case for disabling
        demos_to_load = None
    if demos_to_load and not os.path.exists(demos_to_load):
        print(f"Warning: Specified demo file '{demos_to_load}' not found. Will not load demos.")
        demos_to_load = None # Don't attempt to load if file doesn't exist

    train_sb3(
        env_id=args.env_id,
        total_timesteps=args.timesteps,
        log_dir=session_log_dir, # Pass the session-specific log dir for finding/saving resume.zip
                                 # The Monitor and CheckpointCallback will use subdirs if log_dir is passed to them,
                                 # but SB3 DQN uses tensorboard_log for its own logging, which needs to be specific.
                                 # So, model.load/save uses main_model_dir/RESUME_MODEL_FILENAME
                                 # model tensorboard_log uses session_log_dir
                                 # Monitor uses session_log_dir implicitly as it's passed to env which is then passed to Monitor
                                 # CheckpointCallback uses session_log_dir explicitly
        load_demos_file=demos_to_load, 
        enable_debug_save_frames=args.save_frames,
        enable_debug_print_actions=args.print_actions_env,
        enable_debug_live_view=args.live_view,
        countdown_seconds=args.countdown,
        device=args.device, # Pass parsed device argument
        action_log_interval=args.action_log_interval, # Pass new arg for callback
        n_stack_frames=args.n_stack_frames, # Pass new arg for frame stacking
        continue_training_new_phase=args.new_phase_from_existing_weights # Pass the new flag
    )

