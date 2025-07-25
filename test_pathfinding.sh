#!/bin/bash

# Configurable directories
INPUT_DIR="data"
INTERMEDIATE_DIR="data/intermediate"
OUTPUT_DIR="output"

# Check if example name was provided
# if [ $# -eq 0 ]; then
#     echo "Usage: $0 <example_name> [example_name ...]"
#     echo "Example: $0 Example1 Example2 Example3"
#     exit 1
# fi

fs=("Example1" "Example2" "Example3")
# fs=("Example3")

# Process each example
for example in "${fs[@]}"; do
# for example in "$@"; do
    echo "Processing $example..."
    
    # Run preprocessing
    echo "Running preprocessing for $example..."
    python -c "from preprocessing.processor import Processor; Processor.process_input('${INPUT_DIR}/${example}.png', '${INPUT_DIR}/${example}_constraints.json', '${INTERMEDIATE_DIR}/${example}_processed.json')"
    
    # Check if preprocessing was successful
    if [ $? -ne 0 ]; then
        echo "❌ Preprocessing failed for $example"
        echo
        continue  # Skip to next example if preprocessing fails
    fi
    
    # Run pathfinding and visualization
    python run_pathfinding.py "$example" "$INTERMEDIATE_DIR" "$OUTPUT_DIR"
    
    # Check if pathfinding was successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed $example"
    else
        echo "❌ Pathfinding failed for $example"
    fi
    
    echo
done

echo "All operations completed"
