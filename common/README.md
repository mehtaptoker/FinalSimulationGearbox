# GearRL Common Data Models

This module defines core data structures used throughout the GearRL project. These models provide a consistent interface for representing mechanical systems, constraints, and validation results.

## Constraints
Represents system constraints for gear layout generation.

**Properties:**
- `torque_ratio`: Desired input/output torque ratio (e.g., "1:2")
- `mass_space_ratio`: Target ratio of gear mass to available space (0-1)
- `boundary_margin`: Minimum clearance from boundary edges (mm)
- `min_gear_size`: Minimum acceptable gear diameter (mm)
- `max_gear_size`: Maximum acceptable gear diameter (mm)

**Example:**
```python
constraints = Constraints(
    torque_ratio="1:2",
    mass_space_ratio=0.7,
    boundary_margin=10.0,
    min_gear_size=15,
    max_gear_size=50
)
```

**Target JSON Structure:**
```json
{
    "torque_ratio": "1:2",
    "mass_space_ratio": 0.7,
    "boundary_margin": 10.0,
    "min_gear_size": 15,
    "max_gear_size": 50
}
```

## Point
Represents a 2D coordinate point.

**Properties:**
- `x`: X-coordinate
- `y`: Y-coordinate

**Example:**
```python
point = Point(84.0, 151.0)
```

**Target JSON Structure:**
```json
{"x": 84.0, "y": 151.0}
```

## Boundary
Defines the system boundary as a polygon.

**Properties:**
- `points`: List of Point objects defining the polygon vertices

**Example:**
```python
boundary = Boundary([
    Point(455.0, 5.0),
    Point(5.0, 100.0),
    Point(4.0, 198.0),
    Point(455.0, 198.0)
])
```

**Target JSON Structure:**
```json
[
    {"x": 455.0, "y": 5.0},
    {"x": 5.0, "y": 100.0},
    {"x": 4.0, "y": 198.0},
    {"x": 455.0, "y": 198.0}
]
```

## Gear
Represents a single gear in the system.

**Properties:**
- `center`: Position of gear center (Point)
- `teeth`: Number of teeth
- `module`: Gear module (mm/tooth)
- `diameter` (computed): Gear diameter (teeth × module)

**Example:**
```python
gear = Gear(
    center=Point(100.0, 150.0),
    teeth=30,
    module=2.0
)
```

**Target JSON Structure:**
```json
{
    "center": {"x": 100.0, "y": 150.0},
    "teeth": 30,
    "module": 2.0
}
```

## GearLayout
Represents a collection of gears in a system.

**Properties:**
- `gears`: List of Gear objects

**Example:**
```python
layout = GearLayout([
    Gear(Point(100.0, 150.0), 20, 1.5),
    Gear(Point(200.0, 150.0), 30, 1.5)
])
```

**Target JSON Structure:**
```json
[
    {
        "center": {"x": 100.0, "y": 150.0},
        "teeth": 20,
        "module": 1.5
    },
    {
        "center": {"x": 200.0, "y": 150.0},
        "teeth": 30,
        "module": 1.5
    }
]
```

## SystemDefinition
Represents the complete system definition.

**Properties:**
- `boundary`: System boundary (Boundary)
- `input_shaft`: Input shaft position (Point)
- `output_shaft`: Output shaft position (Point)
- `constraints`: System constraints (Constraints)

**Example:**
```python
system = SystemDefinition(
    boundary=boundary,
    input_shaft=Point(84.0, 151.0),
    output_shaft=Point(393.0, 108.0),
    constraints=constraints
)
```

**Target JSON Structure:**
```json
{
    "boundary_poly": [
        {"x": 455.0, "y": 5.0},
        {"x": 5.0, "y": 100.0},
        {"x": 4.0, "y": 198.0},
        {"x": 455.0, "y": 198.0}
    ],
    "input_shaft": {"x": 84.0, "y": 151.0},
    "output_shaft": {"x": 393.0, "y": 108.0},
    "constraints": {
        "torque_ratio": "1:2",
        "mass_space_ratio": 0.7,
        "boundary_margin": 10.0,
        "min_gear_size": 15,
        "max_gear_size": 50
    }
}
```

## ValidationReport
Represents the result of system validation.

**Properties:**
- `is_valid`: Overall validation status
- `errors`: List of validation error messages

**Example:**
```python
report = ValidationReport(
    is_valid=False,
    errors=["Gears overlap by 2.5mm", "Torque ratio mismatch: 1:1.8 vs required 1:2"]
)
```

**Target JSON Structure:**
```json
{
    "is_valid": false,
    "errors": [
        "Gears overlap by 2.5mm",
        "Torque ratio mismatch: 1:1.8 vs required 1:2"
    ]
}
