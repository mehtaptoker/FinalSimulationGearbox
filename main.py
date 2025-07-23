import argparse
import os
import sys
import json
from pathlib import Path
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from preprocessing.processor import Processor
from pathfinding.finder import Pathfinder
from rl_agent.agents.ppo_agent import PPOAgent as PPORLAgent
from visualization.renderer import Renderer
from common.data_models import SystemDefinition

def main(input_name: str):
    # Validate input files
    input_image = Path(f"data/{input_name}.png")
    input_constraints = Path(f"data/{input_name}_constraints.json")
    
    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")
    if not input_constraints.exists():
        raise FileNotFoundError(f"Input constraints not found: {input_constraints}")

    # Preprocessing
    processed_path = Path(f"data/intermediate/{input_name}_processed.json")
    Processor.process_input(input_image, input_constraints, processed_path)
    
    # Pathfinding
    pathfinder = Pathfinder()
    pathfinder.find_path(processed_path)
    
    # Load processed data
    with open(processed_path, 'r') as f:
        processed_data = json.load(f)
    
    # RL Agent
    agent = PPORLAgent()
    gear_layout = agent.design_gear_layout(
        obstacles=processed_data["obstacles"],
        entry_point=processed_data["entry_point"],
        exit_point=processed_data["exit_point"],
        constraints=processed_data["constraints"]
    )
    
    # Create system definition
    system = SystemDefinition(
        input_name=input_name,
        constraints=processed_data["constraints"],
        gear_layout=gear_layout
    )
    
    # Save final design
    output_dir = Path(f"outputs/{input_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    system_output = output_dir / "system.json"
    with open(system_output, 'w') as f:
        json.dump(system.to_dict(), f, indent=2)
    
    # Render final design
    render_output = output_dir / "system.png"
    renderer = Renderer()
    renderer.render_system(system, render_output)
    
    print(f"✅ Successfully generated system for {input_name}")
    print(f"  - System definition: {system_output}")
    print(f"  - Visualization: {render_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate gear system from input image and constraints'
    )
    parser.add_argument('--input_name', required=True, 
                        help='Base name of input files (without extension)')
    args = parser.parse_args()
    
    try:
        main(args.input_name)
    except Exception as e:
        print(f"❌ Error processing {args.input_name}: {str(e)}", file=sys.stderr)
        sys.exit(1)
