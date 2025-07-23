# Gear Generator Module

The Gear Generator module provides functionality for creating gear objects with proper geometric properties based on mechanical engineering principles.

## API Reference

### GearFactory Class

#### `__init__(self, base_module: float = 1.0)`
Initializes the gear factory with a base module size.

**Parameters:**
- `base_module`: Standard size for gear teeth (default 1.0)

#### `create_gear(self, center: Point, teeth: int) -> Gear`
Creates a gear with specified center and number of teeth.

**Parameters:**
- `center`: Center point of the gear (x, y)
- `teeth`: Number of teeth on the gear

**Returns:**
- Gear object with calculated properties

**Raises:**
- `ValueError` if teeth count is less than 8 or more than 200

### Gear Properties
- `center`: Point object representing gear center
- `teeth`: Number of teeth
- `module`: Gear module size (calculated based on tooth count)
- `diameter`: Pitch diameter (teeth × module)

### Usage Example

```python
from gear_generator.factory import GearFactory
from common.data_models import Point

factory = GearFactory()
gear = factory.create_gear(Point(10, 10), 24)
print(f"Gear created: {gear.teeth} teeth, module={gear.module}, diameter={gear.diameter}")
```

## Dependencies
- Python standard library
- `common.data_models` module
