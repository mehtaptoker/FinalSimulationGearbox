import json
import sys
import numpy as np

import os
from preprocessing.processor import Processor
from gear_generator.factory import GearFactory
from geometry_env.simulator import GearTrainSimulator
from common.data_models import Gear, Point 
from visualization.renderer import Renderer
from pathfinding.finder import Pathfinder


def test_run_full_pipeline(fn='Example1'):
    """
    Runs the full pipeline: preprocessing and then gear generation simulation.
    """
    # 1. --- Configuration ---
    # Define directories and the base name of the example to test.
    CONFIG = {
        "INPUT_DIR": "../data",
        "INTERMEDIATE_DIR": "../data/intermediate",
        "EXAMPLE_NAME": f"{fn}",
        "module": 1.0,
        "clearance_margin": 1.0,
        "initial_gear_teeth": 20,
        "OUTPUT_DIR": "../output",
    }

    # 2. --- Preprocessing Step ---
    print(f"--- Running Preprocessing for: {CONFIG['EXAMPLE_NAME']} ---")
    os.makedirs(CONFIG["INTERMEDIATE_DIR"], exist_ok=True)
    input_img_path = os.path.join(CONFIG["INPUT_DIR"], f"{CONFIG['EXAMPLE_NAME']}.png")
    input_constraints_path = os.path.join(CONFIG["INPUT_DIR"], f"{CONFIG['EXAMPLE_NAME']}_constraints.json")
    processed_json_path = os.path.join(CONFIG["INTERMEDIATE_DIR"], f"{CONFIG['EXAMPLE_NAME']}_processed.json")
    
    Processor.process_input(input_img_path, input_constraints_path, processed_json_path)
    print(f"Successfully generated processed file: {processed_json_path}")

    # 3. --- Pathfinding Step ---
    print("\n--- Finding Optimal Path ---")
    path_json_path = os.path.join(CONFIG['OUTPUT_DIR'], 'path.json')
    path_image_path = os.path.join(CONFIG['OUTPUT_DIR'], 'path.png')
    
    finder = Pathfinder()
    optimal_path = finder.find_path(processed_json_path)
    
    if optimal_path:
        # Save path to JSON
        with open(path_json_path, 'w') as f:
            json.dump(optimal_path, f, indent=4)
        print(f"Path saved to {path_json_path}")

        # Generate visualization
        path_points = [Point(x=p[0], y=p[1]) for p in optimal_path]
        Renderer.render_processed_data(
            processed_json_path,
            path_image_path,
            path=path_points
        )
        print(f"Visualization saved to {path_image_path}")
    else:
        print("FATAL: Could not find a path between shafts.")
    
    # 4. --- Initialization ---
    print("\n--- Initializing Gear Generation Test ---")
    gear_factory = GearFactory(module=CONFIG["module"])
    with open(processed_json_path, 'r') as f:
        data = json.load(f)['normalized_space']
        path_data = data['boundaries']
        shaft_input = tuple(data['input_shaft'].values())
        shaft_output = tuple(data['output_shaft'].values())

    simulator = GearTrainSimulator(
        path=optimal_path, # <-- Use the generated optimal path
        input_shaft=shaft_input,
        output_shaft=shaft_output,
        boundaries=data['boundaries'], # Boundaries are still used for collision checks
        gear_factory=gear_factory,
        clearance_margin=CONFIG["clearance_margin"]
    )

    # 5. --- Execution: Place Input and Intermediate Gears ---
    simulator.reset(initial_gear_teeth=CONFIG["initial_gear_teeth"])
    print(f"Step 0: Initial gear placed (ID: {simulator.last_gear.id}).")

    # Define actions for the two intermediate gears.
    # Action 1: A compound gear (40 driven teeth, 15 driving teeth).
    # Action 2: A simple gear (30 driven teeth, 30 driving teeth).
    intermediate_actions = [
        (10, 15),  # Action 1: Place a compound gear
        (20, 15),  # Action 2: Place another compound gear
        (30, 30)   # Action 3: Place a simple gear
    ]
    
    done = False
    for i, action in enumerate(intermediate_actions):
        print("-" * 20)
        print(f"Step {i+1}: Placing intermediate gear with teeth {action}...")
        state, reward, done, info = simulator.step(action)
        
        if done:
            print(f"Simulation failed trying to place intermediate gear: {info.get('error')}")
            break
        else:
            print(f"Step {i+1}: Intermediate gear placed (ID: {simulator.last_gear.id}).")

    # 6. --- Execution: Calculate and Place Final Gear (only if intermediate steps succeeded) ---
    if not done:
        print("-" * 20)
        print("Step 3: Calculating and placing final gear on output shaft...")
        last_intermediate_gear = simulator.last_gear
        
        dist_to_output = np.linalg.norm(np.array(last_intermediate_gear.center) - np.array(shaft_output))
        final_gear_radius = dist_to_output - last_intermediate_gear.driving_radius
        
        if final_gear_radius * 2 < (8 * CONFIG["module"]): # Check if the gear would be too small
             print(f"Final gear placement failed: Required radius ({final_gear_radius:.2f}) is too small.")
        else:
            final_gear_diameter = final_gear_radius * 2
            final_gear = gear_factory.create_gear_from_diameter(
                gear_id='gear_final',
                center=shaft_output,
                desired_diameter=final_gear_diameter
            )
            simulator.gears.append(final_gear)
            print("Step 3: Final gear placed.")

    # 7. --- Save Generated Gears to JSON ---
    print("\n--- Saving generated gear train to JSON ---")
    output_dir = os.path.join(CONFIG["OUTPUT_DIR"], CONFIG["EXAMPLE_NAME"])
    os.makedirs(output_dir, exist_ok=True)
    gear_layout_path = os.path.join(output_dir, "gear_layout.json")

    # Convert gear objects to JSON-serializable dictionaries
    gears_json_data = [gear.to_json() for gear in simulator.gears]
    
    with open(gear_layout_path, 'w') as f:
        json.dump(gears_json_data, f, indent=4)
    print(f"Gear layout saved to: {gear_layout_path}")

    # 8. --- Final Visualization ---
    print("\n--- Generating final visualization from saved files ---")
    output_image_path = os.path.join(output_dir, "gear_train_result.png")
    
    # The renderer now reads the gear layout from the JSON file
    with open(gear_layout_path, 'r') as f:
        gears_data = json.load(f)

    gears_for_renderer = [Gear.from_json(g) for g in gears_data]

    Renderer.render_processed_data(
        processed_data_path=processed_json_path,
        output_path=output_image_path,
        path=simulator.path,
        gears=gears_for_renderer
    )
    print(f"Visualization saved to: {output_image_path}")

if __name__ == "__main__":
    test_run_full_pipeline("Example3")
