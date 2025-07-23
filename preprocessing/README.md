# Preprocessing Module

The preprocessing module handles conversion of raw input files (images and JSON constraints) into a structured, normalized intermediate representation.

## Processor Class

The `Processor` class provides static methods for processing input files:

### `process_input(png_path, constraints_path, output_path, config=None)`

Processes input files and generates intermediate representation.

**Parameters:**
- `png_path`: Path to input PNG image
- `constraints_path`: Path to JSON constraints file  
- `output_path`: Path for output JSON file
- `config`: Optional configuration dictionary

**Returns:**  
Loaded constraints object

**Processing Steps:**
1. Loads constraints from JSON
2. Processes image to detect:
   - Red input shaft (center coordinates)
   - Green output shaft (center coordinates)
   - Boundary contour (approximated as polygon)
3. Normalizes all coordinates to [-50, 50] range
4. Saves processed data to JSON file

## Output JSON Structure

The output JSON contains three main sections:

1. `pixel_space`: Original coordinates in pixel space
2. `normalized_space`: Normalized coordinates in [-50,50] range
3. `normalization_params`: Parameters used for normalization

Example structure:
```json
{
  "pixel_space": {
    "boundaries": [[x1,y1], [x2,y2], ...],
    "input_shaft": {"x": x, "y": y},
    "output_shaft": {"x": x, "y": y}
  },
  "normalized_space": {
    "boundaries": [[x1,y1], [x2,y2], ...],
    "input_shaft": {"x": x, "y": y},
    "output_shaft": {"x": x, "y": y}
  },
  "normalization_params": {
    "scale": factor,
    "offset_x": offset,
    "offset_y": offset
  }
}
```

All coordinates in `normalized_space` are guaranteed to be within [-50, 50] range.
