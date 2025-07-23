# Gear Generator Academic Methods Documentation

## Parametric Gear Modeling

The GearFactory class implements parametric gear modeling based on standard mechanical engineering principles for spur gears. This documentation covers the mathematical foundations and design considerations.

### Gear Terminology

1. **Module (m)**: Fundamental gear parameter representing size of teeth
   - `m = p / π` where p is circular pitch
2. **Pitch Diameter (d)**: Diameter of pitch circle where teeth mesh
   - `d = m × z` where z is number of teeth
3. **Addendum**: Radial distance from pitch circle to tooth tip
   - Standard addendum = 1 module
4. **Dedendum**: Radial distance from pitch circle to tooth root
   - Standard dedendum = 1.25 modules

### Module Selection Algorithm

The module sizing follows industry-standard practices based on tooth count:

```math
m = 
\begin{cases} 
0.75 \times m_{\text{base}} & \text{if } z \leq 20 \\
1.0 \times m_{\text{base}} & \text{if } 21 \leq z \leq 40 \\
1.25 \times m_{\text{base}} & \text{if } 41 \leq z \leq 60 \\
1.5 \times m_{\text{base}} & \text{if } 61 \leq z \leq 80 \\
2.0 \times m_{\text{base}} & \text{if } z \geq 81 \\
\end{cases}
```

### Design Constraints

1. **Minimum Teeth (8)**:
   - Prevents undercutting in standard pressure angle gears
   - Ensures proper meshing and load distribution
2. **Maximum Teeth (200)**:
   - Practical manufacturing limits
   - Avoids excessive size and inertia

### Mathematical Foundations

1. **Tooth Profile**:
   - Based on involute curve geometry
   - Generated using parametric equations:
     ```math
     x = r_b (\cos\theta + \theta \sin\theta)
     ```
     ```math
     y = r_b (\sin\theta - \theta \cos\theta)
     ```
   - Where rb is base circle radius

2. **Gear Meshing**:
   - Law of gearing requires constant angular velocity ratio
   - Achieved through conjugate action of involute profiles

### Validation Methods
1. Tooth count validation (8 ≤ z ≤ 200)
2. Module calculation verification
3. Pitch diameter consistency check
