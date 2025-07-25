import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from typing import List, Tuple, Dict, Any
from common.data_models import SystemDefinition, Boundary, Point, Constraints, Gear
import numpy as np
import json

class Renderer:
    @staticmethod
    def render_system(system: SystemDefinition, output_path: str, 
                      path: List[Tuple[float, float]] = None, 
                      gears: List[Gear] = None) -> None:
        """
        Render a gear system, including outlines of generated gears and their IDs.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot boundary polygon
        boundary_points = [(p.x, p.y) for p in system.boundary.points]
        boundary_poly = Polygon(boundary_points, closed=True, fill=False, color='black', linewidth=2)
        ax.add_patch(boundary_poly)
        
        # Plot generated gears if provided
        if gears:
            for gear in gears:
                # For compound gears, find the largest radius for annotation positioning
                radii = [d / 2 for d in gear.diameters]
                largest_radius = max(radii)
                
                # Draw a circle outline for each gear in the set
                for radius in radii:
                    gear_circle = Circle(
                        (gear.center.x, gear.center.y), radius,
                        fill=False, # Draw as an outline
                        edgecolor='darkblue', linewidth=1.5
                    )
                    ax.add_patch(gear_circle)
                
                # Add a dot for the center of the gear set
                ax.plot(gear.center.x, gear.center.y, 'ko', markersize=3)

                # Add the gear ID text annotation on the edge of the largest circle
                # Place text at a 45-degree angle from the center
                angle_rad = np.deg2rad(45)
                text_x = gear.center.x + (largest_radius + 1.0) * np.cos(angle_rad) # Small offset
                text_y = gear.center.y - (largest_radius + 1.0) * np.sin(angle_rad) # Y is inverted
                ax.text(text_x, text_y, gear.id, fontsize=9, ha='center', va='center', color='darkgreen')

        # Plot input and output shafts
        ax.plot(system.input_shaft.x, system.input_shaft.y, 'ro', markersize=10, label='Input Shaft')
        ax.plot(system.output_shaft.x, system.output_shaft.y, 'bo', markersize=10, label='Output Shaft')
        
        # Plot path if provided
        if path:
            path_x = [p.x for p in path]
            path_y = [p.y for p in path]
            ax.plot(path_x, path_y, 'm-', linewidth=1, alpha=0.7, label='Path')
        
        # Set plot properties
        all_x = [p[0] for p in boundary_points]
        all_y = [p[1] for p in boundary_points]
        ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
        ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_title('Gear System Visualization')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    @staticmethod
    def render_processed_data(processed_data_path: str, output_path: str, 
                              path: List[Tuple[float, float]] = None, 
                              gears: List[Gear] = None) -> None:
        """
        Render processed data from JSON, optionally including a path and generated gears.
        """
        with open(processed_data_path, 'r') as f:
            data = json.load(f)
            
        norm_space = data['normalized_space']
        
        # Create a minimal SystemDefinition from the processed data
        system = SystemDefinition(
            boundary=Boundary(points=[Point(x=p[0], y=p[1]) for p in norm_space['boundaries']]),
            input_shaft=Point(x=norm_space['input_shaft']['x'], y=norm_space['input_shaft']['y']),
            output_shaft=Point(x=norm_space['output_shaft']['x'], y=norm_space['output_shaft']['y']),
            constraints=None # Constraints are not needed for rendering
        )
        
        # Render the system with optional path and gears
        Renderer.render_system(system, output_path, path, gears)
