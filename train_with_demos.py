import numpy as np
import pickle
import tensorflow as tf
from DQNCNN import DQNCNNAgent
from gym.envs.aidan_envs.geom_wars import GeomEnv

def load_demos(filename):
    """Load demonstrations from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def preprocess_demos(demos):
    """Convert demonstrations into a format suitable for training."""
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    for demo in demos:
        for transition in demo:
            states.append(transition['state'])
            actions.append(transition['action'])
            rewards.append(transition['reward'])
            next_states.append(transition['next_state'])
            dones.append(transition['done'])
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'next_states': np.array(next_states),
        'dones': np.array(dones)
    }

def behavioral_cloning(agent, demo_data, epochs=10, batch_size=32):
    """Train the agent using behavioral cloning."""
    print("Starting behavioral cloning...")
    
    # Convert actions to indices for training
    action_indices = np.argmax(agent.action_to_index(demo_data['actions']), axis=1)
    
    # Train the model
    history = agent.model.fit(
        demo_data['states'],
        action_indices,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    return history

def fine_tune(agent, env, demo_data, episodes=1000, epsilon_start=0.1, epsilon_end=0.01):
    """Fine-tune the agent using DQN with demonstration data in replay buffer."""
    print("Starting fine-tuning...")
    
    # Add demonstration data to replay buffer
    for i in range(len(demo_data['states'])):
        agent.store(
            demo_data['states'][i],
            demo_data['actions'][i],
            demo_data['rewards'][i],
            demo_data['next_states'][i],
            demo_data['dones'][i]
        )
    
    # Training loop
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / episodes
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.store(state, action, reward, next_state, done)
            
            # Train on batch
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        # Update epsilon
        epsilon = max(epsilon_end, epsilon - epsilon_decay)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
        
        # Save model periodically
        if (episode + 1) % 100 == 0:
            agent.save(f"model_episode_{episode + 1}.h5")

def main():
    # Load demonstrations
    print("Loading demonstrations...")
    demos = load_demos("demos.pkl")  # Update with your demo file
    demo_data = preprocess_demos(demos)
    
    # Create environment and agent
    env = GeomEnv()
    agent = DQNCNNAgent(env)
    
    # Behavioral cloning
    print("Training with behavioral cloning...")
    history = behavioral_cloning(agent, demo_data)
    
    # Save pre-trained model
    agent.save("pretrained_model.h5")
    
    # Fine-tune with DQN
    print("Fine-tuning with DQN...")
    fine_tune(agent, env, demo_data)
    
    # Save final model
    agent.save("final_model.h5")

if __name__ == "__main__":
    main() 