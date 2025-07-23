#!/bin/bash

# Check if example name was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <example_name> [example_name ...]"
    echo "Example: $0 Example1 Example2 Example3"
    exit 1
fi

# Process each example
for example in "$@"; do
    echo "Processing $example..."
    
    # Run pathfinding and visualization
    python run_pathfinding.py "$example"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed $example"
    else
        echo "❌ Failed to process $example"
    fi
    
    echo
done

echo "All operations completed"
