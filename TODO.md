# Project TODO List

To train an RL agent to get a high score in Geometry Wars: Retro Evolved from visual input and imitation learning.

## Migration to Gymnasium and Stable Baselines3

*   [x] **Install Libraries:**
    *   Added `gymnasium` and `stable-baselines3` to `requirements.txt`.
    *   Successfully installed packages in the virtual environment.
    *   **Status:** Completed.

*   [x] **Refactor Custom Environment (`geom_wars_env`):**
    *   [x] Rename old `gym` directory to `geom_wars_env` (initially `custom_gym_env`).
    *   [x] Restructure `geom_wars_env` to have `envs/` and `utils/` subdirectories.
        *   [x] `geom_wars_env/utils/window_input.py` (populated with user's code)
        *   [x] `geom_wars_env/utils/read_memory.py` (populated with user's code)
        *   [x] `geom_wars_env/envs/geom_wars_env_core.py` (populated with `GeomEnv` class)
    *   [x] Clean up `__init__.py` files for proper packaging.
    *   [x] Populate `geom_wars_env/envs/geom_wars_env_core.py` with the `GeomEnv` class logic:
        *   [x] Inherit from `gymnasium.Env`.
        *   [x] Adapt screen capture, action handling, reward calculation from reference `geom_wars.py`.
        *   [x] Ensure `reset()` returns `(obs, info)` and `step()` returns `(obs, reward, terminated, truncated, info)`.
        *   [x] Update internal imports to use `..utils.window_input` and `..utils.read_memory`.
    *   [x] Register `GeomEnv` in `geom_wars_env/envs/__init__.py` with `gymnasium.envs.registration.register`.
    *   **Status:** Completed.

*   [ ] **Update Training Pipeline(s) (e.g., `run.py`, `train_with_demos.py`):**
    *   [ ] Replace custom agent (e.g., `DQNCNNAgent`) with `stable-baselines3` agents (e.g., `DQN`).
    *   [ ] Instantiate the `GeomWarsEnv-v0` environment using `gymnasium.make()`.
    *   [ ] Utilize SB3's `agent.learn()` method for training.
    *   [ ] Adapt any data loading or preprocessing for SB3.
    *   [ ] Update TensorBoard logging to use SB3's logger.
    *   **Status:** Pending.

*   [ ] **Update Imitation Learning (Behavioral Cloning) Pipeline:**
    *   [ ] Adapt recording script (`record_demo.py`):
        *   Ensure it saves data (states, actions, rewards, next_states, terminated/truncated) compatible with SB3 or a chosen imitation learning library/method.
        *   Use the new `geom_wars_env` for interaction.
    *   [ ] Decide on a behavioral cloning strategy with SB3:
        *   Consider `stable_baselines3.common.bc.BC` or DAgger if applicable.
        *   Pre-train an SB3 agent or augment its replay buffer.
    *   **Status:** Pending.

## High Priority (Legacy - Mostly Completed)

*   [x] **Fix Observation Space and Screen Capture Mismatch:**
    *   Relevant logic to be ported to `geom_wars_env/envs/geom_wars_env_core.py`.
    *   **Status:** Completed (pending porting to new structure).

*   [x] **Resolve Input Interference with Host Machine:**
    *   Logic now in `geom_wars_env/utils/window_input.py`.
    *   **Status:** Completed.

*   [x] **Correct Initial State for Agent:**
    *   Relevant logic to be ported to `geom_wars_env/envs/geom_wars_env_core.py`.
    *   **Status:** Completed (pending porting to new structure).

*   [x] **Ensure Correct State Reshaping for CNN:**
    *   Relevant logic to be ported to `geom_wars_env/envs/geom_wars_env_core.py`.
    *   **Status:** Completed (pending porting to new structure).

## Medium Priority (Legacy - Mostly Completed)

*   [x] **Refine Reward System:**
    *   All logic to be ported to `_calculate_reward()` in `geom_wars_env/envs/geom_wars_env_core.py`.
    *   **Status:** Completed (pending porting to new structure).

*   [x] **Centralize Observation Gathering:**
    *   Logic to be ported to `_get_observation()` in `geom_wars_env/envs/geom_wars_env_core.py`.
    *   **Status:** Completed (pending porting to new structure).

## Low Priority - Enhancements & Best Practices (To be revisited with SB3)

*   [ ] **Systematize Training and Evaluation with Stable Baselines3:**
    *   [ ] Implement distinct evaluation mode using SB3's `evaluate_policy` or custom callbacks.
    *   [ ] Plan for systematic hyperparameter tuning (e.g., Optuna integration with SB3).
    *   [ ] Leverage SB3's built-in TensorBoard logging.
    *   [ ] Utilize SB3's model checkpointing callbacks.
    *   **Status:** Pending.

*   [x] **Code Refactoring and Organization:**
    *   [x] Refactored environment code into `geom_wars_env` package.
    *   [ ] Manage configurations using a config file (e.g., YAML, JSON) or command-line arguments (e.g., `argparse`).
    *   [ ] Review and simplify main loop in training scripts after SB3 integration.
    *   **Status:** Partially Completed.

*   [x] **Review `noop` function usage:**
    *   The `noop` function (mapping to `stop(window_handle)`) is part of `geom_wars_env/utils/window_input.py` and used in `geom_wars_env_core.py`.
    *   **Status:** Completed.