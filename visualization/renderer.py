import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from common.data_models import SystemDefinition, Boundary, Gear

class Renderer:
    """Renders gear system designs to PNG images with normalized coordinates"""
    
    def render_system(self, system_definition: SystemDefinition, output_path: str):
        """
        Render a gear system design to a PNG image
        
        Args:
            system_definition: SystemDefinition object containing gear layout and boundaries
            output_path: Path to save the output PNG image
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set axis limits to [-50, 50] as specified
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal')  # Ensure circles appear as circles
        ax.grid(True)
        
        # Render boundary
        boundary = system_definition.boundary
        if boundary:
            poly = Polygon(
                [(v.x, v.y) for v in boundary.vertices],
                closed=True,
                fill=False,
                edgecolor='blue',
                linewidth=2
            )
            ax.add_patch(poly)
        
        # Render gears
        for gear in system_definition.gears:
            circle = Circle(
                (gear.position.x, gear.position.y),
                radius=gear.radius,
                fill=True,
                edgecolor='black',
                facecolor='lightgray',
                alpha=0.7
            )
            ax.add_patch(circle)
            
            # Add gear teeth count
            ax.text(
                gear.position.x,
                gear.position.y,
                str(gear.teeth),
                ha='center',
                va='center',
                fontsize=9,
                fontweight='bold'
            )
        
        # Set title and labels
        ax.set_title(f"Gear System Design - {system_definition.name}")
        ax.set_xlabel("X Coordinate (normalized)")
        ax.set_ylabel("Y Coordinate (normalized)")
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
