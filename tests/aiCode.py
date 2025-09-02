import json
import sys
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.processor import Processor
from gear_generator.factory import GearFactory
from geometry_env.simulator import GearTrainSimulator
from common.data_models import Gear, Point 
from visualization.renderer import Renderer
from pathfinding.finder import Pathfinder

from shapely.geometry import Point as ShapelyPoint, Polygon

def ensure_path(processed_json_path: str, path_json_path: str, path_image_path: str):
    """Load or generate path."""
    os.makedirs(os.path.dirname(path_json_path), exist_ok=True)

    if os.path.exists(path_json_path):
        with open(path_json_path, 'r') as f:
            path = json.load(f)
    else:
        finder = Pathfinder()
        path = finder.find_path(processed_json_path)
        if not path:
            raise RuntimeError("Could not find path")
        with open(path_json_path, 'w') as f:
            json.dump(path, f, indent=2)

    Renderer.render_path(processed_json_path, path_image_path, path=path)
    return path

def is_gear_within_boundaries(gear: Gear, boundaries: list, safety_margin: float = 0.5) -> bool:
    """Check if gear is within boundaries."""
    try:
        gear_polygon = ShapelyPoint(gear.center.x, gear.center.y).buffer(gear.driven_radius + safety_margin)
        
        boundary_coords = []
        for b in boundaries:
            if hasattr(b, 'x') and hasattr(b, 'y'):
                boundary_coords.append((b.x, b.y))
            elif isinstance(b, dict):
                boundary_coords.append((b['x'], b['y']))
            elif isinstance(b, (list, tuple)) and len(b) >= 2:
                boundary_coords.append((b[0], b[1]))
        
        if len(boundary_coords) >= 3:
            boundary_polygon = Polygon(boundary_coords)
            return boundary_polygon.contains(gear_polygon)
    except Exception:
        pass
    
    return True

def calculate_max_radius_for_boundaries(center_point: Point, boundaries: list, safety_margin: float = 1.0) -> float:
    """Calculate maximum radius considering boundaries."""
    try:
        center = ShapelyPoint(center_point.x, center_point.y)
        
        boundary_coords = []
        for b in boundaries:
            if hasattr(b, 'x') and hasattr(b, 'y'):
                boundary_coords.append((b.x, b.y))
            elif isinstance(b, dict):
                boundary_coords.append((b['x'], b['y']))
            elif isinstance(b, (list, tuple)) and len(b) >= 2:
                boundary_coords.append((b[0], b[1]))
        
        if len(boundary_coords) >= 3:
            boundary_polygon = Polygon(boundary_coords)
            distance_to_boundary = center.distance(boundary_polygon.boundary)
            return max(0, distance_to_boundary - safety_margin)
    except Exception:
        pass
    
    return float('inf')

def calculate_proper_meshing_position(current_gear, target_direction, new_gear_radius, clearance):
    """Calculate exact position for proper meshing."""
    current_pos = np.array([current_gear.center.x, current_gear.center.y])
    
    # For proper meshing: center distance = driving radius + driven radius + clearance
    required_distance = current_gear.driving_radius + new_gear_radius + clearance
    
    # Normalize direction
    direction_length = np.linalg.norm(target_direction)
    if direction_length == 0:
        return None
        
    unit_direction = target_direction / direction_length
    new_position = current_pos + unit_direction * required_distance
    
    return new_position

def find_valid_meshing_direction(simulator, current_gear, target_pos, boundaries):
    """Find a direction that allows proper meshing within constraints."""
    current_pos = np.array([current_gear.center.x, current_gear.center.y])
    target_direction = np.array(target_pos) - current_pos
    target_distance = np.linalg.norm(target_direction)
    
    if target_distance == 0:
        return None
        
    primary_direction = target_direction / target_distance
    
    # Try the direct direction first
    directions_to_try = [primary_direction]
    
    # Add alternative directions by rotating around the primary direction
    for angle in [15, -15, 30, -30, 45, -45, 60, -60, 90, -90]:
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                  [np.sin(angle_rad), np.cos(angle_rad)]])
        alt_direction = rotation_matrix @ primary_direction
        directions_to_try.append(alt_direction)
    
    # Test each direction with different gear sizes
    module = simulator.gear_factory.module
    gear_sizes = [12, 10, 8]  # Try different sizes
    
    for direction in directions_to_try:
        for teeth_count in gear_sizes:
            gear_radius = (teeth_count * module) / 2
            
            # Calculate meshing position
            mesh_position = calculate_proper_meshing_position(
                current_gear, direction, gear_radius, simulator.clearance_margin
            )
            
            if mesh_position is None:
                continue
            
            # Check if this position is valid
            try:
                # Create test gear at this position
                if hasattr(simulator.path[0], 'x'):
                    center_point = Point(x=mesh_position[0], y=mesh_position[1])
                else:
                    center_point = (mesh_position[0], mesh_position[1])
                
                test_gear = simulator.gear_factory.create_gear(
                    'test_gear',
                    center_point,
                    teeth_count
                )
                
                # Validate boundaries
                if not is_gear_within_boundaries(test_gear, boundaries, safety_margin=0.8):
                    continue
                
                # Validate collision with existing gears
                collision_free = True
                for existing_gear in simulator.gears:
                    center_distance = np.linalg.norm(
                        np.array([test_gear.center.x, test_gear.center.y]) - 
                        np.array([existing_gear.center.x, existing_gear.center.y])
                    )
                    required_distance = test_gear.driven_radius + existing_gear.driven_radius + simulator.clearance_margin
                    
                    if center_distance < required_distance:
                        collision_free = False
                        break
                
                if collision_free:
                    print(f"    Found valid meshing direction: {teeth_count} teeth at [{mesh_position[0]:.2f}, {mesh_position[1]:.2f}]")
                    return direction, teeth_count, mesh_position
                    
            except Exception as e:
                continue
    
    return None, None, None

def create_proper_meshing_chain(simulator, target_position, max_gears=6):
    """Create a proper meshing gear chain with geometric validation."""
    
    current_gear = simulator.last_gear
    target_pos = np.array(target_position)
    
    print(f"Creating proper meshing chain to {target_pos}")
    
    gears_placed = 0
    
    while gears_placed < max_gears:
        current_pos = np.array([current_gear.center.x, current_gear.center.y])
        remaining_distance = np.linalg.norm(target_pos - current_pos)
        
        print(f"\nStep {gears_placed + 1}: Distance to target: {remaining_distance:.2f}")
        
        if remaining_distance < 25:
            print("  Close enough to target for final connection")
            break
        
        # Find a valid meshing direction and size
        direction, teeth_count, mesh_position = find_valid_meshing_direction(
            simulator, current_gear, target_pos, simulator.boundaries
        )
        
        if direction is None:
            print("  No valid meshing direction found")
            break
        
        # Place the gear using simulator
        try:
            # Set path point for simulator
            if hasattr(simulator.path[0], 'x'):
                temp_point = Point(x=mesh_position[0], y=mesh_position[1])
            else:
                temp_point = [mesh_position[0], mesh_position[1]]
            
            # Temporarily update path
            original_index = simulator.current_path_index
            if simulator.current_path_index < len(simulator.path):
                simulator.path[simulator.current_path_index] = temp_point
            
            action = (teeth_count, teeth_count)
            state, reward, done, info = simulator.step(action)
            
            if done and info.get('error'):
                print(f"  Simulator rejected gear: {info.get('error')}")
                simulator.current_path_index = original_index
                break
            else:
                print(f"  Successfully placed {teeth_count}-tooth gear")
                current_gear = simulator.last_gear
                gears_placed += 1
                
                # Advance path index
                if simulator.current_path_index < len(simulator.path) - 2:
                    simulator.current_path_index += 1
                
        except Exception as e:
            print(f"  Error placing gear: {e}")
            break
    
    print(f"Placed {gears_placed} gears in meshing chain")
    return gears_placed > 0

def test_run_full_pipeline(fn='Example1'):
    """Pure validation-based gear generation with proper meshing."""
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG = {
        "INPUT_DIR": os.path.join(BASE_DIR, "data"),
        "INTERMEDIATE_DIR": os.path.join(BASE_DIR, "data", "intermediate"),
        "EXAMPLE_NAME": f"{fn}",
        "module": 1.0,
        "clearance_margin": 1.0,  # Standard clearance
        "initial_gear_teeth": 20,
        "OUTPUT_DIR": os.path.join(BASE_DIR, "output"),
    }

    # Preprocessing
    print(f"--- Preprocessing {CONFIG['EXAMPLE_NAME']} ---")
    os.makedirs(CONFIG["INTERMEDIATE_DIR"], exist_ok=True)
    input_img_path = os.path.join(CONFIG["INPUT_DIR"], f"{CONFIG['EXAMPLE_NAME']}.png")
    input_constraints_path = os.path.join(CONFIG["INPUT_DIR"], f"{CONFIG['EXAMPLE_NAME']}_constraints.json")
    processed_json_path = os.path.join(CONFIG["INTERMEDIATE_DIR"], f"{CONFIG['EXAMPLE_NAME']}_processed.json")
    
    Processor.process_input(input_img_path, input_constraints_path, processed_json_path)

    # Pathfinding
    print("\n--- Pathfinding ---")
    path_json_path = os.path.join(CONFIG['OUTPUT_DIR'], fn, 'path.json')
    path_image_path = os.path.join(CONFIG['OUTPUT_DIR'], fn, 'path.png')
    
    optimal_path = ensure_path(processed_json_path, path_json_path, path_image_path)
    print(f"Path has {len(optimal_path)} points")

    # Initialization
    print("\n--- Initialization ---")
    gear_factory = GearFactory(module=CONFIG["module"])
    with open(processed_json_path, 'r') as f:
        data = json.load(f)['normalized_space']
        shaft_input = tuple(data['input_shaft'].values())
        shaft_output = tuple(data['output_shaft'].values())

    simulator = GearTrainSimulator(
        path=optimal_path,
        input_shaft=shaft_input,
        output_shaft=shaft_output,
        boundaries=data['boundaries'],
        gear_factory=gear_factory,
        clearance_margin=CONFIG["clearance_margin"]
    )

    simulator.reset(initial_gear_teeth=CONFIG["initial_gear_teeth"])
    print(f"Initial gear: {simulator.last_gear.id} at ({simulator.last_gear.center.x:.2f}, {simulator.last_gear.center.y:.2f})")

    # Create proper meshing chain
    print("\n--- Creating Proper Meshing Chain ---")
    
    success = create_proper_meshing_chain(simulator, shaft_output, max_gears=8)
    
    # Attempt final connection
    print("\n--- Final Connection Attempt ---")
    current_gear = simulator.last_gear
    current_pos = np.array([current_gear.center.x, current_gear.center.y])
    final_distance = np.linalg.norm(np.array(shaft_output) - current_pos)
    
    print(f"Final distance to output: {final_distance:.2f}")
    
    if 10 < final_distance < 50:
        direction_to_output = (np.array(shaft_output) - current_pos) / final_distance
        
        # Calculate what size gear can bridge this gap
        available_radius = (final_distance - current_gear.driving_radius - CONFIG["clearance_margin"]) / 2
        max_teeth = int((available_radius * 2) / CONFIG["module"])
        
        for final_teeth in [max_teeth, max_teeth-2, max_teeth-4]:
            if final_teeth < 8:
                continue
                
            final_radius = (final_teeth * CONFIG["module"]) / 2
            required_distance = current_gear.driving_radius + final_radius + CONFIG["clearance_margin"]
            
            if final_distance >= required_distance * 0.95:  # 95% tolerance
                final_position = current_pos + direction_to_output * required_distance
                
                # Validate final gear position
                try:
                    if hasattr(simulator.path[0], 'x'):
                        temp_point = Point(x=final_position[0], y=final_position[1])
                    else:
                        temp_point = [final_position[0], final_position[1]]
                    
                    test_final_gear = simulator.gear_factory.create_gear(
                        'test_final',
                        temp_point,
                        final_teeth
                    )
                    
                    if is_gear_within_boundaries(test_final_gear, simulator.boundaries):
                        # Try to place it
                        if simulator.current_path_index < len(simulator.path):
                            simulator.path[simulator.current_path_index] = temp_point
                        
                        action = (final_teeth, final_teeth)
                        state, reward, done, info = simulator.step(action)
                        
                        if not (done and info.get('error')):
                            print(f"Final connection successful with {final_teeth} teeth!")
                            break
                except Exception:
                    continue

    # Save and analyze results
    print(f"\n--- Results ---")
    output_dir = os.path.join(CONFIG["OUTPUT_DIR"], CONFIG["EXAMPLE_NAME"])
    os.makedirs(output_dir, exist_ok=True)
    gear_layout_path = os.path.join(output_dir, "gear_layout_proper_meshing.json")

    gears_json_data = [gear.to_json() for gear in simulator.gears]
    with open(gear_layout_path, 'w') as f:
        json.dump(gears_json_data, f, indent=4)
    
    output_image_path = os.path.join(output_dir, "gear_train_proper_meshing.png")
    gears_for_renderer = [Gear.from_json(g) for g in gears_json_data]

    Renderer.render_processed_data(
        processed_data_path=processed_json_path,
        output_path=output_image_path,
        path=simulator.path,
        gears=gears_for_renderer
    )
    
    print(f"Saved {len(simulator.gears)} gears")
    print(f"Proper meshing visualization: {output_image_path}")
    
    # Detailed meshing analysis
    print(f"\n--- Detailed Meshing Analysis ---")
    total_gears = len(simulator.gears)
    proper_meshing_count = 0
    
    for i, gear in enumerate(simulator.gears[1:], 1):
        prev_gear = simulator.gears[i-1]
        distance = np.linalg.norm(np.array([gear.center.x, gear.center.y]) - 
                                np.array([prev_gear.center.x, prev_gear.center.y]))
        expected_distance = gear.driven_radius + prev_gear.driving_radius
        difference = abs(distance - expected_distance)
        
        if difference < 1.5:  # Tolerance for proper meshing
            proper_meshing_count += 1
            print(f"Gear {gear.id}: ✓ PROPER MESHING (distance: {distance:.2f}, expected: {expected_distance:.2f})")
        else:
            print(f"Gear {gear.id}: ✗ Meshing issue (distance: {distance:.2f}, expected: {expected_distance:.2f}, diff: {difference:.2f})")
    
    if total_gears > 1:
        meshing_rate = proper_meshing_count / (total_gears - 1) * 100
        print(f"\nPROPER MESHING RATE: {meshing_rate:.1f}%")
        print(f"TOTAL GEARS: {total_gears}")
        
        if meshing_rate >= 80:
            print("✓ EXCELLENT meshing quality!")
        elif meshing_rate >= 60:
            print("◆ GOOD meshing quality")  
        else:
            print("✗ Poor meshing quality - geometry issues detected")
    else:
        print("Only input/output gears placed - no intermediate chain created")

if __name__ == "__main__":
    test_run_full_pipeline("Example1")