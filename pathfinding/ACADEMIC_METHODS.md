# Pathfinder Academic Methods Documentation

## A* Algorithm Implementation

The Pathfinder module implements the A* (A-Star) pathfinding algorithm to find the shortest path between two points while avoiding obstacles. A* is a best-first search algorithm that uses heuristics to guide its search efficiently.

### Algorithm Overview

1. **Initialization**:
   - Create open and closed sets
   - Add start node to open set with:
     - g = 0 (cost from start to current node)
     - h = heuristic estimate to end node
     - f = g + h (total estimated cost)

2. **Main Loop**:
   - While open set is not empty:
     - Select node with lowest f-score
     - If current node is target, reconstruct path
     - Move current node to closed set
     - For each neighbor:
       - Skip if in closed set or blocked by obstacle
       - Calculate tentative g-score
       - If better path found, update neighbor's scores and path

3. **Path Reconstruction**:
   - Backtrack from target node using parent pointers
   - Reverse path for start-to-target order

### Key Components

**Heuristic Function**:  
Uses Euclidean distance between points:
`h(n) = √((n.x - target.x)² + (n.y - target.y)²)`

**Neighbor Generation**:  
Generates 8-direction neighbors (N, S, E, W, NE, NW, SE, SW)

**Obstacle Avoidance**:  
Checks if neighbor positions intersect with boundary polygons

### Complexity
- Time: O(b^d) where b is branching factor, d is solution depth
- Space: O(b^d) for storing open/closed sets

### Optimization Considerations
1. **Heuristic Quality**: Euclidean distance is optimal for grid movement
2. **Data Structures**: Uses dictionaries for O(1) lookups
3. **Boundary Checking**: Simple point-in-polygon check for obstacles
