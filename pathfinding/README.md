# Pathfinding Module

## Pathfinder API

The `Pathfinder` class implements the A* algorithm for finding paths between input and output shafts in normalized coordinate space.

### Class Definition
```python
class Pathfinder:
    def find_path(self, processed_data_path: str) -> list:
        ...
```

### Method: `find_path(processed_data_path)`
- **Arguments**:
  - `processed_data_path` (str): Path to processed JSON file containing normalized space data
- **Returns**:
  - List of [x, y] points representing the path in normalized space, or None if no path exists

### Input Data Structure
The input JSON file must contain a `normalized_space` object with:
- `boundaries`: List of [x,y] points defining workspace boundaries
- `input_shaft`: {x, y} coordinates of input shaft
- `output_shaft`: {x, y} coordinates of output shaft

Example structure:
```json
{
  "normalized_space": {
    "boundaries": [[x1,y1], [x2,y2], ...],
    "input_shaft": {"x": x_val, "y": y_val},
    "output_shaft": {"x": x_val, "y": y_val}
  }
}
```

### Usage Example
```python
from pathfinding.finder import Pathfinder

finder = Pathfinder()
path = finder.find_path("data/intermediate/Example1_processed.json")

if path:
    print(f"Path found with {len(path)} points")
else:
    print("No path found")
