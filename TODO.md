# Project TODO List

## High Priority - Core Functionality & Stability

*   [x] **Fix Observation Space and Screen Capture Mismatch:**
    *   The `observation_space` defined in `gym/envs/aidan_envs/geom_wars.py` now uses consistent dimensions (100x100).
    *   Screen capture logic has been moved into the environment's `_get_observation()` method.
    *   Window dimensions and capture region are properly calculated and cached.
    *   **Status:** Completed - Observation space and screen capture are now consistent.

*   [x] **Resolve Input Interference with Host Machine:**
    *   Created new `window_input.py` module using `win32gui.PostMessage` for window-specific input.
    *   Implemented proper window handle management.
    *   **Status:** Completed - Input is now window-specific and doesn't interfere with host machine.

*   [x] **Correct Initial State for Agent:**
    *   `GeomEnv.reset()` now properly captures and returns initial screen state.
    *   Screen capture logic centralized in `_get_observation()` method.
    *   **Status:** Completed - Initial state is now properly captured and returned.

*   [x] **Ensure Correct State Reshaping for CNN:**
    *   Observation space updated to include channel dimension: `shape=(STATEX, STATEY, 1)`.
    *   `_get_observation()` now returns properly shaped grayscale images.
    *   **Status:** Completed - State reshaping is now correct for CNN input.

## Medium Priority - RL Mechanics & Training Loop

*   [x] **Refine Reward System:**
    *   [x] **Implement Explicit Survival Reward:**
        *   Added `SURVIVAL_REWARD` constant for step-by-step survival.
        *   Added `LIFE_LOST_PENALTY` for life loss.
    *   [x] **Correct Time-Based Reward Logic:**
        *   Implemented proper time-based rewards with `TIME_REWARD_INTERVAL`.
        *   Added `steps_since_last_time_reward` tracking.
    *   [x] **Balance Rewards:** Implemented balanced reward system with:
        *   Survival rewards
        *   Life loss penalties
        *   Score-based rewards
        *   Time-based rewards
    *   [x] **Centralize Reward Calculation:** Moved all reward logic to `_calculate_reward()` in environment.

*   [x] **Centralize Observation Gathering:**
    *   Moved screen capture to environment's `_get_observation()`.
    *   Added window caching and error handling.
    *   Added frame skipping for better performance.
    *   **Status:** Completed - Observation gathering is now centralized and robust.

*   [ ] **Implement Imitation Learning (Behavioral Cloning):**
    *   [ ] Create a script to record human gameplay (states, actions, rewards, next_states, dones).
        *   Leverage existing screen capture from `run.py` and keyboard input mapping.
    *   [ ] Decide on a strategy:
        *   Pre-train the `DQNCNNAgent` using the human data.
        *   Augment the agent's replay buffer with human demonstrations.

## Low Priority - Enhancements & Best Practices

*   [ ] **Systematize Training and Evaluation:**
    *   [ ] Implement a distinct evaluation mode for the agent.
    *   [ ] Plan for systematic hyperparameter tuning.
    *   [ ] Enhance TensorBoard logging.
    *   [ ] Implement model checkpointing.

*   [ ] **Code Refactoring and Organization:**
    *   [ ] Manage configurations using a config file or command-line arguments.
    *   [ ] Review and simplify the main loop in `run.py`.

*   [x] **Review `noop` function usage:**
    *   The `noop` function now correctly maps to `stop(window_handle)`.
    *   **Status:** Completed - No-op behavior is now properly implemented.