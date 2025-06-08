import gymnasium
import geom_wars_env # This should trigger the registration
env = gymnasium.make('GeomWarsEnv-v0')
print("Environment created successfully!")
obs, info = env.reset()
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
env.close()