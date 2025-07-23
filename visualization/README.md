# Visualization Module

This module provides rendering capabilities for gear system designs using normalized coordinates. The primary component is the `Renderer` class which generates PNG images of gear layouts.

## Installation

Install required dependencies:
```bash
pip install matplotlib numpy
```

## Usage

### Basic Example
```python
from visualization.renderer import Renderer
from common.data_models import SystemDefinition, Boundary, Point, Gear

# Create a system definition
boundary = Boundary(
    vertices=(
        Point(-40, -40),
        Point(40, -40),
        Point(40, 40),
        Point(-40, 40)
    )
)

gears = [
    Gear(position=Point(0, 0), radius=10, teeth=20),
    Gear(position=Point(20, 20), radius=8, teeth=16)
]

system = SystemDefinition(
    name="Sample System",
    boundary=boundary,
    gears=gears
)

# Render to PNG
renderer = Renderer()
renderer.render_system(system, "output.png")
```

### Renderer Class
The `Renderer` class has one main method:

#### `render_system(system_definition: SystemDefinition, output_path: str)`
Renders a gear system design to a PNG image.

**Parameters:**
- `system_definition`: SystemDefinition object containing:
  - `boundary`: Boundary object with vertices
  - `gears`: List of Gear objects
  - `name`: System name (used in title)
- `output_path`: Path to save the output PNG file

**Output Features:**
- Axis limits fixed to [-50, 50] in both dimensions
- Equal aspect ratio to prevent distortion
- Boundary rendered as blue polygon
- Gears rendered as semi-transparent circles
- Gear teeth count displayed at gear center
- Grid lines for reference
- Title with system name
- Axis labels

## Examples
![Sample Output](data/intermediate/Example1_system.png)

## Testing
Run unit tests:
```bash
python -m unittest tests/test_renderer.py
```

Tests cover:
- Systems with boundary and gears
- Systems with boundary only
- Systems with gears only
- Empty systems

## Integration
This module can be integrated with other components to visualize:
- Initial designs from preprocessing
- Optimized designs from RL agent
- Validation results from physics validator
