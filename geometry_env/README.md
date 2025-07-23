# Environment Manager

The `EnvironmentManager` class manages the geometric environment including obstacles and defines fixed input/output shaft locations.

## API Documentation

### `INPUT_SHAFT` and `OUTPUT_SHAFT`
Class attributes representing fixed shaft positions:
```python
INPUT_SHAFT = Point(x=0.0, y=0.0)
OUTPUT_SHAFT = Point(x=10.0, y=0.0)
```

### `load_obstacles_from_json(file_path: str) -> List[Boundary]`
Loads obstacle data from a JSON file and converts it into Boundary objects.

**Parameters**:
- `file_path` (str): Path to JSON file containing obstacle definitions

**Returns**:
- List[Boundary]: List of Boundary objects representing obstacles

**Raises**:
- `FileNotFoundError`: If specified file doesn't exist
- `ValueError`: If JSON structure is invalid or obstacle definitions are malformed

## Example Usage

```python
from geometry_env.manager import EnvironmentManager

# Load obstacles from JSON file
obstacles = EnvironmentManager.load_obstacles_from_json("path/to/obstacles.json")

# Access fixed shaft locations
input_shaft = EnvironmentManager.INPUT_SHAFT
output_shaft = EnvironmentManager.OUTPUT_SHAFT

print(f"Loaded {len(obstacles)} obstacles")
print(f"Input shaft at: ({input_shaft.x}, {input_shaft.y})")
print(f"Output shaft at: ({output_shaft.x}, {output_shaft.y})")
```

## JSON Format Requirements

Obstacle files must be JSON with the following structure:
```json
{
  "obstacles": [
    [
      {"x": 0.0, "y": 0.0},
      {"x": 10.0, "y": 0.0},
      {"x": 10.0, "y": 10.0}
    ],
    [
      {"x": 20.0, "y": 20.0},
      {"x": 30.0, "y": 20.0},
      {"x": 25.0, "y": 30.0}
    ]
  ]
}
```

- Each obstacle must be a polygon with at least 3 points
- Each point must have both x and y coordinates
