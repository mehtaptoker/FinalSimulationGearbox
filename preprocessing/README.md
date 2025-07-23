# Preprocessing Module

## Overview
The preprocessing module converts raw input files (images and JSON constraints) into structured intermediate representations. This standardized format enables consistent processing by downstream system components.

## Processor Class
The `Processor` class provides the core functionality for input conversion.

### Methods
#### `process_input(png_path, constraints_path, output_path, config=None)`
Processes input files and generates intermediate representation.

**Parameters**:
- `png_path`: Path to input PNG image
- `constraints_path`: Path to JSON constraints file
- `output_path`: Path for output JSON file
- `config`: Optional configuration dictionary (default: None)

**Returns**:
Loaded constraints object for immediate use by main script

**Workflow**:
1. Loads constraints from JSON file
2. Processes PNG image:
   - Detects red (input) and green (output) shafts
   - Isolates and approximates boundary contours
3. Creates structured intermediate representation
4. Saves representation as JSON file
5. Returns loaded constraints

### Intermediate File Format
The generated JSON file contains:
```json
{
  "boundaries": [[x1,y1], [x2,y2], ...],
  "input_shaft": {"x": x, "y": y},
  "output_shaft": {"x": x, "y": y}
}
```

## Usage Example
```python
from preprocessing.processor import Processor

constraints = Processor.process_input(
    "data/Example1.png",
    "data/Example1_constraints.json",
    "data/intermediate/Example1_processed.json"
)
