import time
import os
import json
import argparse
import numpy as np
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_env.env import GearEnv
from rl_agent.agents.ppo_agent import PPOAgent 

def main():
    parser = argparse.ArgumentParser(description='Train RL agent for gear train generation')
    # --- Arguments ---
    parser.add_argument('--config_path', type=str, required=True, help='Path to the environment config JSON file.')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=10, help='Maximum steps per episode')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for optimizer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--log_interval', type=int, default=20, help='Log progress every N episodes')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Environment Setup ---
    with open(args.config_path, 'r') as f:
        env_config = json.load(f)
    
    env = GearEnv(env_config)

    # --- Agent Setup ---
    # Get state and action dimensions directly from the environment
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec # For MultiDiscrete, this gives a list of choices, e.g., [33, 33]

    # Your custom PPOAgent must be designed to handle a MultiDiscrete action space
    agent = PPOAgent(
        state_dim=state_dim,
        action_dims=action_dims, # Pass the list of action dimensions
        lr=args.learning_rate,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon
    )

    print("--- Starting Agent Training ---")
    start_time = time.time()

    # --- Training Loop ---
    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(args.max_steps):
            # Agent selects an action
            action, log_prob = agent.act(state)
            
            # Environment executes the action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience in agent's memory
            agent.buffer.add(state, action, reward, done, log_prob)

            state = next_state
            episode_reward += reward

            if done:
                break
        
        # Agent updates its policy
        loss = agent.update()

        if episode % args.log_interval == 0:
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Loss: {loss:.4f}")

    end_time = time.time()
    print(f"--- Training Finished in {end_time - start_time:.2f}s ---")
    
    # Save the final model
    model_path = os.path.join(args.output_dir, "ppo_gear_placer_final.pt")
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()

if __name__ == "__main__":
    main()
