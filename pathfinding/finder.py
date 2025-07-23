import json
import math
from heapq import heappop, heappush

class Pathfinder:
    """
    Finds a path between two points within a polygon using a grid-based A* search.
    """
    def find_path(self, processed_data_path):
            # 1. Load and parse the data
            with open(processed_data_path, 'r') as f:
                data = json.load(f)['normalized_space']
            
            boundaries = [tuple(p) for p in data['boundaries']]
            start_node = (data['input_shaft']['x'], data['input_shaft']['y'])
            end_node = (data['output_shaft']['x'], data['output_shaft']['y'])
            
            # 2. Verify start/end points are inside the container
            if not self._is_inside(start_node, boundaries):
                print("Error: Start point is outside the boundary container.")
                return None
            if not self._is_inside(end_node, boundaries):
                print("Error: End point is outside the boundary container.")
                return None
                
            # 3. Initialize A* search with consistent, rounded keys
            step_size = 0.5
            start_node_key = (round(start_node[0], 4), round(start_node[1], 4))

            open_set = [(0, start_node)] # Heap stores (f_score, original_node_tuple)
            came_from = {}
            
            g_score = {start_node_key: 0}
            f_score = {start_node_key: self._distance(start_node, end_node)}

            while open_set:
                _, current = heappop(open_set)
                
                # Create a rounded key for dictionary lookups
                current_key = (round(current[0], 4), round(current[1], 4))

                # Check if we have reached the goal
                if self._distance(current, end_node) < step_size:
                    path = self._reconstruct_path(came_from, current)
                    path.append(list(end_node))
                    return path

                # Explore neighbors
                for neighbor in self._get_neighbors(current, boundaries, end_node, step_size):
                    # Always use the rounded key to access g_score
                    tentative_g_score = g_score[current_key] + self._distance(current, neighbor)
                    
                    neighbor_key = (round(neighbor[0], 4), round(neighbor[1], 4))

                    if tentative_g_score < g_score.get(neighbor_key, float('inf')):
                        came_from[neighbor_key] = current
                        g_score[neighbor_key] = tentative_g_score
                        f_score[neighbor_key] = tentative_g_score + self._distance(neighbor, end_node)
                        heappush(open_set, (f_score[neighbor_key], neighbor))
            
            print("No path found after exploring all possibilities.")
            return None

    def _get_neighbors(self, point, boundaries, goal, step):
        """Generates valid neighbors for a point."""
        neighbors = []
        
        # Optimization: Check for a direct line of sight to the goal
        if self._has_line_of_sight(point, goal, boundaries, step):
            neighbors.append(goal)
            return neighbors # If we can see the goal, that's the only neighbor we need

        # Standard 8-directional grid neighbors
        for dx in [-step, 0, step]:
            for dy in [-step, 0, step]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (point[0] + dx, point[1] + dy)
                if self._is_inside(neighbor, boundaries):
                    neighbors.append(neighbor)
        return neighbors

    def _has_line_of_sight(self, p1, p2, boundaries, resolution):
        """
        Approximates line of sight by checking intermediate points.
        Returns True if all points on the line segment are inside the polygon.
        """
        dist = self._distance(p1, p2)
        if dist < resolution:
            return True
        
        num_checks = int(dist / resolution)
        dx = (p2[0] - p1[0]) / num_checks
        dy = (p2[1] - p1[1]) / num_checks

        for i in range(1, num_checks):
            intermediate_point = (p1[0] + i * dx, p1[1] + i * dy)
            if not self._is_inside(intermediate_point, boundaries):
                return False
        return True

    def _is_inside(self, point, boundaries):
        """Checks if a point is inside a polygon using the Ray Casting algorithm."""
        x, y = point
        n = len(boundaries)
        inside = False
        p1x, p1y = boundaries[0]
        for i in range(n + 1):
            p2x, p2y = boundaries[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= x_intersect:
                                inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _distance(self, p1, p2):
        """Calculates Euclidean distance."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _reconstruct_path(self, came_from, current):
        """Builds the final path from the A* result."""
        path = [list(current)]
        current_key = (round(current[0], 4), round(current[1], 4))
        while current_key in came_from:
            current = came_from[current_key]
            current_key = (round(current[0], 4), round(current[1], 4))
            path.append(list(current))
        path.reverse()
        return path