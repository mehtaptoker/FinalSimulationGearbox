import argparse
import time
import torch
import numpy as np
from rl_agent.agents.ppo_agent import PPOAgent
from common.data_models import ValidationReport, Constraints
import math

def compute_reward(
    report: ValidationReport, 
    constraints: Constraints, 
    target_torque: float,
    torque_weight: float,
    space_weight: float,
    weight_penalty_coef: float
) -> float:
    """
    Calculate reward based on validation report and constraints
    
    Args:
        report: Validation report from physics validator
        constraints: Design constraints
        target_torque: Desired torque ratio
        torque_weight: Weight for torque component
        space_weight: Weight for space usage component
        weight_penalty_coef: Weight penalty coefficient
        
    Returns:
        float: Calculated reward scalar
    """
    # Heavy penalty for invalid designs (collisions, etc)
    if not report.is_valid:
        return -10.0  # Significant penalty for invalid designs
    
    # Calculate torque reward (exponential decay for closeness to target)
    torque_diff = abs(report.torque_ratio - target_torque)
    torque_reward = math.exp(-torque_diff)  # [0,1] range
    
    # Calculate space usage reward (higher = better)
    space_reward = report.space_usage
    
    # Calculate weight penalty (lower mass = better)
    weight_penalty = report.total_mass * 0.01  # Scale mass penalty
    
    # Weighted sum of components
    reward = (
        torque_weight * torque_reward + 
        space_weight * space_reward - 
        weight_penalty_coef * weight_penalty
    )
    return reward

def main():
    parser = argparse.ArgumentParser(description='Train RL agent for mechanical design optimization')
    # Existing arguments
    parser.add_argument('--agent', type=str, default='ppo', choices=['ppo'], help='RL agent type')
    parser.add_argument('--constraints_file', type=str, required=True, help='Path to constraints JSON file')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum steps per episode')
    parser.add_argument('--save_interval', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--log_interval', type=int, default=10, help='Log progress every N episodes')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models')
    
    # Algorithm-specific hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate for optimizer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    
    # Reward function parameters
    parser.add_argument('--target_torque', type=float, default=2.0, help='Target torque ratio for reward calculation')
    parser.add_argument('--torque_weight', type=float, default=0.6, help='Weight for torque component in reward')
    parser.add_argument('--space_weight', type=float, default=0.3, help='Weight for space usage component in reward')
    parser.add_argument('--weight_penalty', type=float, default=0.1, help='Weight penalty coefficient for mass')
    
    args = parser.parse_args()

    # Load constraints to get environment boundaries
    constraints = Constraints.load_from_file(args.constraints_file)
    
    # Calculate state dimension based on environment
    # State tensor includes:
    # - Gear configuration: position (2), radius (1), teeth (1), rotation (1) -> 5 per gear
    # - Boundary vertices: 2 coordinates per vertex (4 vertices) -> 8
    # - Obstacles: position (2), size (2) per obstacle -> 4 per obstacle
    # - System state: torque ratio (1), space usage (1), mass (1) -> 3
    max_gears = 10  # Maximum number of gears we can have in a design
    max_obstacles = len(constraints.obstacles) if constraints.obstacles else 0
    state_dim = (max_gears * 5) + 8 + (max_obstacles * 4) + 3
    
    # Action space definition:
    # 0: No-op
    # 1-4: Add gear (x, y, radius, teeth)
    # 5-6: Remove gear (index)
    # 7-9: Adjust position (index, x, y)
    # 10-12: Change size (index, radius, teeth)
    action_dim = 13  # Based on action space defined in ACADEMIC_METHODS.md
    
    if args.agent == 'ppo':
        agent = PPOAgent(
            state_dim, 
            action_dim,
            lr=args.learning_rate,
            gamma=args.gamma,
            clip_epsilon=args.clip_epsilon,
            entropy_coef=args.entropy_coef
        )
    else:
        raise ValueError(f"Unsupported agent type: {args.agent}")

    best_reward = -float('inf')
    best_design = None
    
    # Training loop
    for episode in range(args.episodes):
        # TODO: Initialize environment with constraints
        # state = env.reset()
        state = np.random.randn(state_dim)  # Placeholder state
        
        episode_reward = 0
        for step in range(args.max_steps):
            # Agent selects action
            action, _ = agent.act(state)
            
            # TODO: Execute action in environment
            # next_state, reward, done, info = env.step(action)
            next_state = np.random.randn(state_dim)  # Placeholder
            
            # Placeholder validation report - will be replaced by actual
            placeholder_report = ValidationReport(
                is_valid=True,
                torque_ratio=1.5 + np.random.rand(),  # Random value near target
                space_usage=np.random.rand(),          # Random space usage
                total_mass=10.0 + np.random.rand() * 5 # Random mass
            )
            reward = compute_reward(
                placeholder_report,
                constraints,  # Need to load constraints
                args.target_torque,
                args.torque_weight,
                args.space_weight,
                args.weight_penalty
            )
            done = (step == args.max_steps - 1)  # Placeholder
            
            # Store experience (simplified - actual PPO uses batches)
            agent.memory.append((state, action, reward, next_state, done))
            
            # Track best design
            if reward > best_reward:
                best_reward = reward
                # TODO: Replace with actual design from environment
                # best_design = info['design']
                best_design = {
                    "episode": episode,
                    "step": step,
                    "reward": reward,
                    "torque_ratio": placeholder_report.torque_ratio,
                    "space_usage": placeholder_report.space_usage,
                    "total_mass": placeholder_report.total_mass
                }
            
            # Update agent
            if done:
                states, actions, rewards, next_states, dones = zip(*agent.memory)
                loss = agent.update(states, actions, rewards, next_states, dones)
                agent.memory = []
                break
            
            state = next_state
            episode_reward += reward
        
        # Logging
        if episode % args.log_interval == 0:
            print(f"Episode {episode}/{args.episodes}, Reward: {episode_reward:.2f}, Loss: {loss:.4f}")
        
        # Save model
        if episode % args.save_interval == 0:
            model_path = f"{args.output_dir}/{args.agent}_episode_{episode}.pt"
            agent.save(model_path)
            print(f"Saved model to {model_path}")
    
    # Save best design to system.json
    if best_design:
        import json
        with open('system.json', 'w') as f:
            json.dump(best_design, f, indent=2)
        print(f"Saved best design to system.json with reward: {best_reward:.2f}")

if __name__ == "__main__":
    main()
