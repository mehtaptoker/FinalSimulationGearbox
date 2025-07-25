#!/bin/bash

# This script evaluates a trained PPO agent model.

echo "--- Evaluating Trained Gear Generation Agent ---"

# Define the path to the model you want to evaluate
# This should point to one of the models saved during training.
MODEL_TO_EVALUATE="models/ppo_gear_placer_final.pt"

# Check if the model file exists
if [ ! -f "$MODEL_TO_EVALUATE" ]; then
    echo "Error: Model file not found at $MODEL_TO_EVALUATE"
    echo "Please make sure you have trained the agent first by running train.sh"
    exit 1
fi

# Execute the evaluation script
python rl_agent/evaluate.py \
    --model_path "$MODEL_TO_EVALUATE" \
    --config_path "env_config.json" \
    --output_dir "output_eval"

echo "--- Evaluation Complete. Check the 'output_eval' directory for results. ---"
