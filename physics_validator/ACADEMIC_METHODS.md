# Physics Validator Academic Methods

This document details the mathematical foundations and algorithms used for physics validation of gear layouts in normalized coordinate space.

## 1. Gear Collision Detection

### Algorithm: Circle-Circle Intersection
Validates that no gears overlap by treating each gear as a circle with diameter = teeth × module.

**Mathematical Formulation:**
For two gears with centers C₁(x₁,y₁), C₂(x₂,y₂) and diameters d₁, d₂:
- Distance between centers: δ = √[(x₂-x₁)² + (y₂-y₁)²]
- Minimum separation: δ_min = (d₁/2) + (d₂/2)
- Collision occurs when: δ < δ_min

**Implementation:**
```python
for i, gear1 in enumerate(gears):
    for j, gear2 in enumerate(gears[i+1:], start=i+1):
        dist = math.sqrt((gear1.center.x - gear2.center.x)**2 + 
                         (gear1.center.y - gear2.center.y)**2)
        min_dist = (gear1.diameter/2) + (gear2.diameter/2)
        if dist < min_dist:
            # Collision detected
```

## 2. Boundary Containment

### Algorithm 1: Point-in-Polygon Test
Verifies gear centers are within boundary polygon using ray casting algorithm.

**Mathematical Formulation:**
- Cast ray from point to infinity (rightward)
- Count polygon edge crossings
- Odd count → inside; even count → outside

**Implementation:**
```python
n = len(polygon)
inside = False
p1 = polygon[0]
for i in range(n + 1):
    p2 = polygon[i % n]
    if point.y > min(p1.y, p2.y):
        if point.y <= max(p1.y, p2.y):
            if point.x <= max(p1.x, p2.x):
                if p1.y != p2.y:
                    x_inters = (point.y-p1.y)*(p2.x-p1.x)/(p2.y-p1.y)+p1.x
                if p1.x == p2.x or point.x <= x_inters:
                    inside = not inside
    p1 = p2
```

### Algorithm 2: Minimum Distance to Boundary
Calculates minimum distance from gear center to any boundary edge.

**Mathematical Formulation:**
Distance from point P to line segment AB:
1. Project P onto AB → P'
2. If P' between A and B: distance = |PP'|
3. Else: distance = min(|PA|, |PB|)

**Implementation:**
```python
min_dist = float('inf')
for i in range(len(boundary)):
    p1 = boundary[i]
    p2 = boundary[(i+1) % len(boundary)]
    dist = point_to_line_distance(point, p1, p2)
    if dist < min_dist:
        min_dist = dist
```

## 3. Torque Ratio Validation

### Algorithm: Gear Ratio Calculation
Validates torque ratio using gear teeth count.

**Mathematical Formulation:**
- Torque ratio = (Output torque)/(Input torque) ≈ (Input teeth)/(Output teeth)
- Target ratio is provided as string (e.g., "3:1" → 3.0)

**Implementation:**
```python
# Find nearest gears to input/output shafts
input_gear = min(gears, key=lambda g: distance(g.center, input_shaft))
output_gear = min(gears, key=lambda g: distance(g.center, output_shaft))

# Calculate actual ratio
actual_ratio = input_gear.teeth / output_gear.teeth

# Compare with target ratio (with 5% tolerance)
if not math.isclose(actual_ratio, target_value, rel_tol=0.05):
    return f"Torque ratio mismatch"
```

## 4. Assumptions and Limitations

1. **Simplified Physics:** 
   - Treats gears as perfect circles
   - Ignores friction, material properties, and dynamic effects
   - Assumes gear mesh occurs when centers are exactly (d₁/2 + d₂/2) apart

2. **Boundary Margin:**
   - Margin is applied uniformly around boundary
   - Doesn't account for irregular clearance requirements

3. **Torque Ratio:**
   - Simplified to input/output gear ratio
   - Doesn't validate intermediate gear relationships
   - Limited to direct input/output gear pairs

## 5. References

1. O'Rourke, J. (1998). *Computational Geometry in C* (2nd ed.). Cambridge University Press.
2. Schneider, P. J., & Eberly, D. H. (2003). *Geometric Tools for Computer Graphics*. Morgan Kaufmann.
3. Norton, R. L. (2020). *Machine Design: An Integrated Approach* (6th ed.). Pearson.
