# Pathfinder Module

The Pathfinder module implements pathfinding functionality using the A* algorithm to find routes between input and output shafts while avoiding obstacles.

## API Reference

### Pathfinder Class

#### `find_path(processed_data_path: str) -> List[Tuple[float, float]]`

Finds a path from input shaft to output shaft using A* algorithm.

**Parameters:**
- `processed_data_path`: Path to JSON file containing:
  - `boundaries`: List of obstacle polygons (each polygon is a list of points)
  - `input_shaft`: Starting point [x, y]
  - `output_shaft`: Target point [x, y]

**Returns:**
- List of points representing the path from input to output shaft

**Raises:**
- `ValueError` if no valid path is found

### Usage Example

```python
from pathfinding.finder import Pathfinder

finder = Pathfinder()
path = finder.find_path("data/intermediate/Example1.json")
print("Path found:", path)
```

## Dependencies
- Python standard libraries: `json`, `math`, `typing`
