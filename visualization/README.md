# Visualization Module

The visualization module provides tools for rendering gear system layouts to image files.

## Renderer Class

The `Renderer` class provides static methods for visualizing gear systems.

### Methods

#### `render_system(system: SystemDefinition, output_path: str) -> None`
Renders a gear system to a PNG file.

**Parameters:**
- `system`: A SystemDefinition object containing the gear layout and boundaries
- `output_path`: Path to save the output PNG file

**Example Usage:**
```python
from visualization.renderer import Renderer
from common.data_models import SystemDefinition, Boundary, Point, Constraints

# Create a sample system
boundary = Boundary(points=[
    Point(x=-50, y=-50),
    Point(x=50, y=-50), 
    Point(x=50, y=50),
    Point(x=-50, y=50)
])
system = SystemDefinition(
    boundary=boundary,
    input_shaft=Point(x=-30, y=0),
    output_shaft=Point(x=30, y=0),
    constraints=Constraints(...)
)

# Render to file
Renderer.render_system(system, "output.png")
```

## Dependencies
- matplotlib
- numpy
