# Reinforcement Learning Agent for Game AI

## Overview

This project is an exploration into reinforcement learning (RL), aiming to train an AI agent from scratch to play a game. The primary inputs for the agent are visual information from the game, the current game score, and the number of lives remaining. The goal is to see if the agent can learn to play the game effectively based on these inputs.

## Core Concept

The agent is being developed using the following principles:

*   **Model:** A Deep Q-Network (DQN) with a Convolutional Neural Network (CNN) to process visual inputs. This is evident from the `DQNCNNAgent` used in the training script (`run.py`, line 26, 43).
*   **Inputs:**
    *   Visual screen captures of the game (handled by libraries like `cv2`, `mss`, and `pyautogui` imported in `run.py`, lines 3, 31, 5).
    *   Game Score: Read directly from the game's memory (using functions like `getScore` from `gym/utils/read_memory.py`, lines 55-58).
    *   Number of Lives: Also read from game memory (e.g., `getLives` from `gym/utils/read_memory.py`, line 23).
*   **Actions:** The agent controls the game by simulating keyboard presses (as seen in `gym/utils/directkeys.py`, lines 1-35, and used by the custom game environment).
*   **Environment:** A custom OpenAI Gym environment appears to be set up, likely for a game similar to Geometry Wars, as suggested by `gym/envs/aidan_envs/geom_wars.py` (lines 1-20).

## Current Status & Challenges

*   **Initial Progress:** The agent seems to be learning and partially working, but there's uncertainty whether more extensive training time is the primary need for better performance.
*   **Input Interference:** A significant challenge is that the agent directly controls the host machine's keyboard (`gym/utils/directkeys.py`, `pyautogui` in `run.py`). This causes conflicts when the Python script and the user are both trying to use the keyboard, leading to "havoc."

## Next Steps & Ideas

1.  **Imitation Learning / Behavioral Cloning:**
    *   **Goal:** To provide the agent with examples of good gameplay, record episodes of human play. This can help bootstrap the learning process or fine-tune the agent.
    *   **Implementation:** Potentially using Gym's built-in monitoring tools (like `gym.wrappers.Monitor` seen in `run.py`, line 42) or custom recording scripts. The `gym/wrappers/monitoring/video_recorder.py` (lines 174-234) might also offer relevant utilities.

2.  **Reward Shaping:**
    *   **Current:** The reward is primarily based on the game score.
    *   **Proposed Change:** Introduce a reward component for survival. For instance, the agent could receive a small positive reward for every N seconds it stays alive.
    *   **Challenge:** Carefully balancing the score-based reward with the survival reward to ensure the agent optimizes for both objectives effectively.

3.  **Environment Isolation / Input Handling:**
    *   **Problem:** The agent's keyboard inputs directly affect the host machine.
    *   **Goal:** Find a method to send inputs to the game without interfering with the host system, or run the game in an isolated environment.
    *   **Potential Avenues to Explore:**
        *   **Virtual Machines (VMs):** Run the game and agent within a VM.
        *   **Window-Specific Input Libraries:** Investigate libraries that can send keystrokes directly to a specific application window without requiring it to be in focus (this can be tricky).
        *   **Headless Environments:** For some games or emulators, it might be possible to run them in a headless mode (no visible GUI) and interact programmatically.
        *   **Using Standardized Gym Wrappers for Emulated Games:** If the game could be run in an emulator that has a stable Gym interface (e.g., like Atari games via `gym/envs/atari/atari_env.py`, lines 1-164), this often handles input/output more cleanly. However, the current setup with `read_process_memory` (`gym/utils/read_memory.py`, lines 40-52) suggests interaction with a native PC game.

## Key Technologies & Components

*   **Programming Language:** Python
*   **Reinforcement Learning Framework:** OpenAI Gym (`gym`)
*   **RL Algorithm:** Deep Q-Network (DQN) with a Convolutional Neural Network (CNN)
*   **Visual Processing:** OpenCV (`cv2`)
*   **Screen Capture:** `mss`, `PIL` (Pillow)
*   **System Automation (for input):** `pyautogui`
*   **Custom Game Environment:** Likely `gym/envs/aidan_envs/geom_wars.py`
    *   Utilizes screen scraping for observations.
    *   Reads game state (score, lives) via memory inspection (`gym/utils/read_memory.py`).
    *   Simulates keyboard inputs for actions (`gym/utils/directkeys.py`).
*   **Training Script:** `run.py` (lines 1-54) orchestrates the training process.

