import os
import json
import sys
from pathfinding.finder import Pathfinder
from visualization.renderer import Renderer

# Get arguments
if len(sys.argv) < 2:
    print("Usage: python run_pathfinding.py <example_name> [intermediate_dir] [output_dir]")
    print("Default: intermediate_dir='data/intermediate', output_dir='output'")
    sys.exit(1)
    
example_name = sys.argv[1]
intermediate_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/intermediate'
output_dir = sys.argv[3] if len(sys.argv) > 3 else 'output'

# Create example-specific output directory
example_output_dir = os.path.join(output_dir, example_name)
os.makedirs(example_output_dir, exist_ok=True)

# Input and output paths
processed_path = os.path.join(intermediate_dir, f'{example_name}_processed.json')
path_json_path = os.path.join(example_output_dir, 'path.json')
image_path = os.path.join(example_output_dir, 'path.png')

# Run pathfinding
finder = Pathfinder()
# path = finder.find_path(processed_path)
path = finder.find_centerline_path(processed_path)

if path:
    # Save path to JSON
    with open(path_json_path, 'w') as f:
        json.dump(path, f, indent=4)
    print(f"Path saved to {path_json_path}")

    # Generate visualization
    Renderer.render_path(
        processed_path,
        image_path,
        path=path
    )
    print(f"Visualization saved to {image_path}")
else:
    print("No path found")
