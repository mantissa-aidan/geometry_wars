# Makes envs a package and handles registration
from gymnasium.envs.registration import register
from .geom_wars_env_core import GeomEnv

register(
    id='GeomWarsEnv-v0',
    entry_point='geom_wars_env.envs.geom_wars_env_core:GeomEnv',
    # Optionally, you can specify max_episode_steps, reward_threshold, etc.
    # max_episode_steps=1000,
)