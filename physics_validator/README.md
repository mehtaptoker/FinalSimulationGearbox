# Physics Validator Module

This module provides physics validation for gear layouts in normalized coordinate space.

## ValidationReport Structure

The ValidationReport class contains:
- `is_valid`: Boolean indicating if the layout meets all constraints
- `errors`: List of descriptive error messages for any violations

## check_layout Method

```python
def check_layout(layout: GearLayout, system: SystemDefinition) -> ValidationReport
```

Validates a gear layout against physical constraints in normalized coordinates.

### Parameters
- `layout`: GearLayout containing all gears in the system
- `system`: SystemDefinition containing boundary and constraints

### Validation Checks
1. **Gear Collisions**: Ensures no gears overlap using circle-circle intersection
2. **Boundary Containment**: Verifies all gears are within boundary polygon with margin
3. **Gear Size**: Validates gear teeth count against min/max constraints
4. **Torque Ratio**: Checks if input/output gear ratio matches target (when specified)

### Return Value
Returns a ValidationReport object with validation results.

## Usage Example

```python
from physics_validator.validator import PhysicsValidator
from common.data_models import GearLayout, SystemDefinition

# Load layout and system definition
layout = GearLayout.from_json(layout_data)
system = SystemDefinition.from_json(system_data)

# Validate layout
report = PhysicsValidator.check_layout(layout, system)

if report.is_valid:
    print("Layout is valid!")
else:
    print("Validation errors:")
    for error in report.errors:
        print(f"- {error}")
