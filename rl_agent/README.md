# Reinforcement Learning Agent Module

This module implements reinforcement learning agents for optimizing mechanical gear system designs. The primary agent is a Proximal Policy Optimization (PPO) agent that learns to generate optimal gear layouts based on physics validation feedback.

## Installation

Ensure you have Python 3.8+ installed. Install dependencies using:

```bash
pip install torch numpy
```

## Training Guide

### Running the Training Script

To train the RL agent, use the following command:

```bash
python train.py --constraints_file path/to/constraints.json [OPTIONS]
```

### Required Arguments
- `--constraints_file`: Path to the JSON file containing design constraints

### Optional Arguments
- `--agent`: RL agent type (default: 'ppo')
- `--episodes`: Number of training episodes (default: 1000)
- `--max_steps`: Maximum steps per episode (default: 200)
- `--save_interval`: Save model every N episodes (default: 100)
- `--log_interval`: Log progress every N episodes (default: 10)
- `--output_dir`: Directory to save models (default: 'models')
- `--learning_rate`: Learning rate for optimizer (default: 0.0003)
- `--gamma`: Discount factor (default: 0.99)
- `--clip_epsilon`: PPO clip parameter (default: 0.2)
- `--entropy_coef`: Entropy coefficient (default: 0.01)
- `--target_torque`: Target torque ratio (default: 2.0)
- `--torque_weight`: Torque component weight in reward (default: 0.6)
- `--space_weight`: Space usage weight in reward (default: 0.3)
- `--weight_penalty`: Mass penalty coefficient (default: 0.1)

### Example Command
```bash
python train.py \
  --constraints_file data/Example1_constraints.json \
  --episodes 2000 \
  --max_steps 300 \
  --learning_rate 0.0001 \
  --output_dir trained_models
```

## Outputs

During training:
- Model checkpoints are saved in the specified output directory
- Training progress is logged to the console

After training:
- The best design is saved to `system.json` containing:
  - Episode and step where the best design was found
  - Reward value
  - Torque ratio
  - Space usage
  - Total mass

## Agent Implementation

The PPO agent implements the following action space:
1. **Add Gear**: Parameters (x, y, radius, teeth)
2. **Remove Gear**: Parameter (gear index)
3. **Adjust Position**: Parameters (index, x, y)
4. **Change Size**: Parameters (index, radius, teeth)

## Reward Function

The reward function is calculated as:

```
R = 
  -10 (for invalid designs) OR
  torque_weight * exp(-|torque_ratio - target_torque|) +
  space_weight * space_usage -
  weight_penalty * (total_mass * 0.01)
```

## Testing

Run unit tests for the reward function:
```bash
python -m unittest tests/test_agent.py
```

## Next Steps
- Integrate with the actual environment implementation
- Implement state representation from environment boundaries
- Add visualization of training progress
- Implement more sophisticated experience replay
