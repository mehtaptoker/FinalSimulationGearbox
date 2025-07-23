import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Tuple
from common.data_models import SystemDefinition, Boundary, Point, Constraints
import numpy as np
import json

class Renderer:
    @staticmethod
    def render_system(system: SystemDefinition, output_path: str, path: List[Tuple[float, float]] = None) -> None:
        """Render a gear system to a PNG file, optionally including a path.
        
        Args:
            system: SystemDefinition containing the gear layout and boundaries
            output_path: Path to save the output PNG file
            path: Optional list of (x, y) points representing a path
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot boundary polygon
        boundary_points = [(p.x, p.y) for p in system.boundary.points]
        boundary_poly = Polygon(boundary_points, closed=True, fill=False, color='black', linewidth=2)
        ax.add_patch(boundary_poly)
        
        # Plot input and output shafts
        ax.plot(system.input_shaft.x, system.input_shaft.y, 'ro', markersize=10, label='Input Shaft')
        ax.plot(system.output_shaft.x, system.output_shaft.y, 'bo', markersize=10, label='Output Shaft')
        
        # Plot path if provided
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'm-', linewidth=2, label='Path')
            ax.plot(path_x, path_y, 'mo', markersize=4)
        
        # Set plot properties
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.invert_yaxis()

        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_title('Gear System Visualization with Path')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    @staticmethod
    def render_processed_data(processed_data_path: str, output_path: str, path: List[Tuple[float, float]] = None) -> None:
        """
        Render processed data from JSON file to an image, optionally including a path.
        
        Args:
            processed_data_path: Path to processed JSON file
            output_path: Path to save output image
            path: Optional list of (x, y) points representing a path
        """
        # Load processed data
        with open(processed_data_path, 'r') as f:
            data = json.load(f)
            
        # Create a minimal SystemDefinition from processed data
        # Use default constraints since they're not in processed data
        default_constraints = Constraints(
            torque_ratio="1:1",
            mass_space_ratio=0.5,
            boundary_margin=0.1,
            min_gear_size=10,
            max_gear_size=100
        )
        
        system = SystemDefinition(
            boundary=Boundary(points=[Point(x=p[0], y=p[1]) for p in data['normalized_space']['boundaries']]),
            input_shaft=Point(x=data['normalized_space']['input_shaft']['x'], 
                              y=data['normalized_space']['input_shaft']['y']),
            output_shaft=Point(x=data['normalized_space']['output_shaft']['x'], 
                               y=data['normalized_space']['output_shaft']['y']),
            constraints=default_constraints
        )
        
        # Render the system with optional path
        Renderer.render_system(system, output_path, path)


