import os
import json
import argparse
import sys
import numpy as np

sys.path.append('../')

from geometry_env.env import GearEnv
from rl_agent.agents.ppo_agent import PPOAgent
from visualization.renderer import Renderer
from common.data_models import Gear 

def evaluate_agent():
    """
    Loads a trained PPO agent and evaluates its performance on the GearEnv
    by running one full episode and visualizing the result.
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained RL agent for gear generation.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained PPO model (.pt file).')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the environment config JSON file.')
    parser.add_argument('--output_dir', type=str, default='output_eval', help='Directory to save evaluation results.')
    args = parser.parse_args()

    # --- Environment and Agent Setup ---
    print("--- Setting up Environment and Agent ---")
    with open(args.config_path, 'r') as f:
        env_config = json.load(f)

    env = GearEnv(env_config)

    # Get state and action dimensions from the environment
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec

    # Instantiate the agent and load the trained weights
    agent = PPOAgent(state_dim, action_dims, lr=0, gamma=0, clip_epsilon=0) # Hyperparams don't matter for eval
    agent.load(args.model_path)

    # --- Run Evaluation Episode ---
    print("\n--- Running Evaluation Episode ---")
    state, _ = env.reset()
    done = False
    episode_reward = 0
    step_count = 0

    while not done:
        # Agent selects the best action deterministically
        action, _ = agent.act(state)
        
        # Environment executes the action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        state = next_state
        episode_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: Action={action}, Reward={reward:.2f}")

    print(f"\n--- Episode Finished ---")
    if info.get("success"):
        print(f"Result: SUCCESS - {info['success']}")
    elif info.get("error"):
        print(f"Result: FAILED - {info['error']}")
    else:
        print("Result: Episode finished due to step limit.")
    print(f"Total Reward: {episode_reward:.2f}")

    # --- Save and Visualize the Result ---
    # Create a unique output directory for this evaluation run
    example_name = os.path.basename(env_config['json_path']).replace('_processed.json', '')
    eval_output_dir = os.path.join(args.output_dir, f"{example_name}_eval")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Save the generated gear layout to a JSON file
    gear_layout_path = os.path.join(eval_output_dir, "evaluation_gear_layout.json")
    gears_json_data = [gear.to_json() for gear in env.simulator.gears]
    with open(gear_layout_path, 'w') as f:
        json.dump(gears_json_data, f, indent=4)
    print(f"\nGenerated gear layout saved to: {gear_layout_path}")

    # Render the final gear train from the saved files
    output_image_path = os.path.join(eval_output_dir, "evaluation_result.png")
    Renderer.render_processed_data(
        processed_data_path=env_config['json_path'],
        output_path=output_image_path,
        path=env.simulator.path,
        gear_layout_path=gear_layout_path
    )
    print(f"Final visualization saved to: {output_image_path}")
    
    env.close()

if __name__ == "__main__":
    evaluate_agent()