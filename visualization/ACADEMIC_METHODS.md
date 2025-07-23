# Visualization of Normalized Mechanical Designs

## Introduction
This document describes the methodology used to translate normalized mechanical design data into 2D graphical representations. The visualization process is crucial for understanding and validating the output of the automated design system.

## Coordinate Normalization
All design elements are represented using normalized coordinates:
- Boundary vertices scaled to [-50, 50] range
- Gear positions within [-50, 50] bounds
- Gear radii proportional to the normalized scale

This normalization allows consistent visualization regardless of the original physical dimensions of the design.

## Rendering Methodology
The visualization process follows these key steps:

1. **Coordinate System Setup**:
   - Fixed axis limits: x ∈ [-50, 50], y ∈ [-50, 50]
   - Equal aspect ratio (1:1) to prevent geometric distortion
   - Grid lines at 10-unit intervals for spatial reference

2. **Boundary Representation**:
   - Boundary vertices connected in sequence to form a closed polygon
   - Blue outline with 2px line width
   - Vertices plotted exactly at their normalized coordinates

3. **Gear Visualization**:
   - Circles centered at (x, y) coordinates with radius r
   - Light gray fill with black border
   - Semi-transparent (alpha=0.7) to show potential overlaps
   - Teeth count displayed at gear center in bold font

4. **Layout and Annotation**:
   - Title with system name
   - Axis labels indicating normalized coordinates
   - Grid for spatial reference
   - Aspect ratio locked to 1:1 to preserve geometric relationships

## Importance of Plot Limits
Setting fixed plot limits to [-50, 50] is critical for:

1. **Consistent Scaling**: Ensures all designs are visualized at the same scale regardless of their specific dimensions.

2. **Accurate Spatial Relationships**: Maintains correct proportions between gears and boundaries.

3. **Comparability**: Allows direct visual comparison between different designs.

4. **Standardized Output**: Creates a consistent visualization framework for all system components.

## Implementation Details
The visualization is implemented using Matplotlib, which provides:
- High-quality vector output for scalability
- Precise control over all visual elements
- Cross-platform compatibility
- Publication-ready figure generation

The renderer creates 1000x1000 pixel PNG images at 150 DPI, providing sufficient detail for analysis while maintaining reasonable file sizes.
