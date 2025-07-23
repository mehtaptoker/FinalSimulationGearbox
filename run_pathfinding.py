import os
import json
import sys
from pathfinding.finder import Pathfinder
from visualization.renderer import Renderer

# Get example name from command line arguments
if len(sys.argv) < 2:
    print("Usage: python run_pathfinding.py <example_name>")
    sys.exit(1)
    
example_name = sys.argv[1]

# Create output directory
output_dir = f'output/{example_name}'
os.makedirs(output_dir, exist_ok=True)

# Run pathfinding
finder = Pathfinder()
path = finder.find_path(f'data/intermediate/{example_name}_processed.json')

if path:
    # Save path to JSON
    path_json_path = os.path.join(output_dir, 'path.json')
    with open(path_json_path, 'w') as f:
        json.dump(path, f, indent=4)
    print(f"Path saved to {path_json_path}")

    # Generate visualization
    image_path = os.path.join(output_dir, 'path.png')
    Renderer.render_processed_data(
        f'data/intermediate/{example_name}_processed.json',
        image_path,
        path=path
    )
    print(f"Visualization saved to {image_path}")
else:
    print("No path found")
