# Environment Representation and Normalization from Multimodal Engineering Inputs

## Methodology

### Image Processing Pipeline

1. **Input Loading**:
   - RGB image loaded using OpenCV
   - Converted to HSV color space for better color separation

2. **Shaft Detection**:
   - **Input Shaft (Red)**:
     - HSV range: [0-10, 120-255, 70-255]
     - Contour detection with RETR_EXTERNAL flag
     - Center calculated using image moments
   - **Output Shaft (Green)**:
     - HSV range: [35-85, 120-255, 70-255]
     - Same processing as input shaft

3. **Boundary Detection**:
   - Convert to grayscale
   - Apply Gaussian blur (5x5 kernel)
   - Edge detection using Canny algorithm (50-150 thresholds)
   - Contour approximation using Ramer-Douglas-Peucker algorithm
     - ε = 0.01 * contour perimeter

### Coordinate Normalization

All geometric features are normalized to a canonical [-50, 50] range:

1. **Bounding Box Calculation**:
   - Find min/max coordinates across all features (shafts + boundary)

2. **Normalization Transform**:
   - Scaling factor (s):
     ```
     s = 100 / max(width, height)
     ```
     where width = max_x - min_x, height = max_y - min_y
   - Translation offsets (t_x, t_y):
     ```
     t_x = -((min_x + max_x) / 2) * s
     t_y = -((min_y + max_y) / 2) * s
     ```

3. **Coordinate Transformation**:
   For each point (x, y):
   ```
   x_norm = x * s + t_x
   y_norm = y * s + t_y
   ```

### Importance of Normalization

1. **Model Generalization**:
   - Removes dependence on absolute image dimensions
   - Ensures consistent scale across different input images
   - Makes learned features scale-invariant

2. **Numerical Stability**:
   - Bounded range prevents extreme values
   - Helps with gradient-based optimization

3. **Physical Interpretation**:
   - Normalized space can be mapped to real-world units
   - Enables consistent physical constraints application
