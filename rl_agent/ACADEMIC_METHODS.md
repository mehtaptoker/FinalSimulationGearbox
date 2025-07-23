# Deep Reinforcement Learning for Combinatorial Optimization in Automated Mechanical Design

## Methodology

### Problem Formulation (Markov Decision Process)

**State Space**: The state representation is derived from the current GearLayout and environment boundaries loaded from the normalized intermediate file (`_processed.json`). The state is converted into a tensor with the following components:

1. **Gear Configuration**: 
   - Position (x, y normalized coordinates)
   - Radius (normalized size)
   - Teeth count
   - Rotation direction (binary flag)
   
2. **Environment Boundaries**:
   - Normalized boundary vertices
   - Obstacle positions and sizes
   - Path constraints

3. **System State**:
   - Current torque ratio
   - Space utilization percentage
   - Total mass of the system

The state tensor is constructed by concatenating these normalized features into a fixed-length vector suitable for neural network input.

**Action Space**: The action space supports an iterative design process within the normalized space with four discrete action types:

1. **Add Gear**: 
   - Parameters: (x, y) position, radius, teeth count
   - Action space: [1, x, y, radius, teeth]

2. **Remove Gear**:
   - Parameters: Gear index to remove
   - Action space: [2, gear_index]

3. **Adjust Position**:
   - Parameters: Gear index, new (x, y) position
   - Action space: [3, gear_index, x, y]

4. **Change Size**:
   - Parameters: Gear index, new radius, new teeth count
   - Action space: [4, gear_index, radius, teeth]

This action space enables fine-grained control over the mechanical design while maintaining the constraints of the normalized space.

**Reward Function**: The reward function is formally defined as:

\[
R(s, a) = \begin{cases} 
-10 & \text{if invalid design} \\
w_t \cdot e^{-|τ - τ_t|} + w_s \cdot u_s - w_m \cdot m & \text{otherwise}
\end{cases}
\]

Where:
- \(τ\) = current torque ratio
- \(τ_t\) = target torque ratio
- \(u_s\) = space utilization (0-1)
- \(m\) = total mass (scaled by 0.01)
- \(w_t\) = torque weight (default 0.6)
- \(w_s\) = space weight (default 0.3)
- \(w_m\) = mass penalty coefficient (default 0.1)

The exponential term for torque reward ensures higher sensitivity near the target torque ratio, while the space utilization and mass terms encourage efficient designs.
