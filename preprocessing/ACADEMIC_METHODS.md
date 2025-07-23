# Environment Representation from Multimodal Engineering Inputs

## Methodology
This document describes the image processing pipeline for converting engineering diagrams into structured environment representations.

### Image Processing Pipeline
1. **Input Acquisition**: Load PNG diagram containing:
   - Mechanical boundary (white outline)
   - Input shaft (red circle)
   - Output shaft (green circle)

2. **Color-space Conversion**:
   - Convert RGB to HSV for robust color detection
   - HSV provides better separation of color components

3. **Shaft Detection**:
   - **Input Shaft (Red)**:
     - HSV range: [0-10, 120-255, 70-255]
     - Morphological operations to reduce noise
     - Contour detection and center calculation using image moments
   - **Output Shaft (Green)**:
     - HSV range: [35-85, 120-255, 70-255]
     - Same processing as input shaft

4. **Boundary Extraction**:
   - Convert to grayscale
   - Apply binary thresholding to isolate boundary
   - Find external contours
   - Approximate contour using Ramer-Douglas-Peucker algorithm:
     - ε = 0.01 * contour perimeter
     - Reduces points while preserving shape

5. **Data Integration**:
   - Combine geometric features with JSON constraints
   - Output structured intermediate representation

### Algorithm Selection
- **Color-based Detection**: Chosen for simplicity and effectiveness with distinct markers
- **Ramer-Douglas-Peucker**: Optimal for polygonal approximation of smooth curves
- **Image Moments**: Provides accurate centroid calculation for circular features

### Validation
- Unit tests verify:
  - Correct shaft positioning (within 2px tolerance)
  - Boundary point count matches expected shape complexity
  - Robust handling of missing/invalid files
- Visual inspection of intermediate files

### Limitations
- Requires distinct color markers (red/green)
- Sensitive to lighting conditions
- Assumes single input/output shafts
