# A* Pathfinding Algorithm Implementation

## Overview
This document details the implementation of the A* (A-Star) pathfinding algorithm used to find optimal paths between input and output shafts in normalized coordinate space.

## Algorithm Pseudocode
```
function A*(start, goal)
    openSet = priority queue containing start
    cameFrom = empty map
    gScore = map with default value of Infinity
    gScore[start] = 0
    fScore = map with default value of Infinity
    fScore[start] = heuristic(start, goal)

    while openSet is not empty
        current = node in openSet with lowest fScore
        if current == goal
            return reconstruct_path(cameFrom, current)

        openSet.remove(current)
        for each neighbor of current
            tentative_gScore = gScore[current] + distance(current, neighbor)
            if tentative_gScore < gScore[neighbor]
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + heuristic(neighbor, goal)
                if neighbor not in openSet
                    openSet.add(neighbor)

    return failure (no path exists)
```

## Implementation Details

### Heuristic Function
The Euclidean distance is used as the heuristic:
```python
def heuristic(self, a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
```

### Node Exploration
- Uses a priority queue (min-heap) to efficiently retrieve the node with the lowest f-score
- Explores neighbors in 8 directions (cardinal and diagonal)
- Step size configurable (default 0.5 units in normalized space)

### Boundary Handling
- Uses ray casting algorithm to determine if points are within workspace boundaries
- Only considers points within boundaries as valid neighbors

### Path Reconstruction
- Backtraces from goal to start using the cameFrom dictionary
- Reverses the path for correct start-to-goal order

## Complexity Analysis
- Time Complexity: O(b^d) where b is branching factor, d is depth
- Space Complexity: O(n) for storing open/closed sets and path data
- Practical performance depends on:
  - Step size (smaller steps → more nodes)
  - Workspace complexity
  - Path length

## Optimization Considerations
1. **Step Size**: Larger steps reduce computation but may miss paths
2. **Heuristic**: Euclidean distance provides good balance for continuous space
3. **Data Structures**: Heap provides efficient min extraction
4. **Boundary Check**: Ray casting is efficient for convex polygons
