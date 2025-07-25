import json
import math
from heapq import heappop, heappush

class Pathfinder:
    """
    Finds a path between two points within a polygon, maintaining a specified
    margin from the boundaries. It uses a grid-based A* search.
    """
    def find_path(self, processed_data_path, margin=5.0):
        # 1. Load and parse the data
        with open(processed_data_path, 'r') as f:
            data = json.load(f)['normalized_space']
        
        boundaries = [tuple(p) for p in data['boundaries']]
        start_node = (data['input_shaft']['x'], data['input_shaft']['y'])
        end_node = (data['output_shaft']['x'], data['output_shaft']['y'])
        
        # 2. Verify start/end points are valid (i.e., inside and respecting the margin)
        if not self._is_valid(start_node, boundaries, margin):
            print(f"Error: Start point is invalid (too close to boundary with margin {margin}).")
            return None
        if not self._is_valid(end_node, boundaries, margin):
            print(f"Error: End point is invalid (too close to boundary with margin {margin}).")
            return None
            
        # 3. Initialize A* search
        step_size = 0.5
        start_node_key = (round(start_node[0], 4), round(start_node[1], 4))

        open_set = [(0, start_node)]
        came_from = {}
        g_score = {start_node_key: 0}
        f_score = {start_node_key: self._distance(start_node, end_node)}

        while open_set:
            _, current = heappop(open_set)
            current_key = (round(current[0], 4), round(current[1], 4))

            if self._distance(current, end_node) < step_size:
                path = self._reconstruct_path(came_from, current)
                path.append(list(end_node))
                return path

            for neighbor in self._get_neighbors(current, boundaries, end_node, step_size, margin):
                tentative_g_score = g_score[current_key] + self._distance(current, neighbor)
                neighbor_key = (round(neighbor[0], 4), round(neighbor[1], 4))

                if tentative_g_score < g_score.get(neighbor_key, float('inf')):
                    came_from[neighbor_key] = current
                    g_score[neighbor_key] = tentative_g_score
                    f_score[neighbor_key] = tentative_g_score + self._distance(neighbor, end_node)
                    heappush(open_set, (f_score[neighbor_key], neighbor))
        
        print("No path found after exploring all possibilities.")
        return None
    
    # New main validity check
    def _is_valid(self, point, boundaries, margin):
        """A point is valid if it's inside and respects the margin."""
        if not self._is_inside(point, boundaries):
            return False
        if self._distance_to_boundary(point, boundaries) < margin:
            return False
        return True

    def _get_neighbors(self, point, boundaries, goal, step, margin):
        """Generates valid neighbors for a point, respecting the margin."""
        neighbors = []
        if self._has_line_of_sight(point, goal, boundaries, step, margin):
            neighbors.append(goal)
            return neighbors

        for dx in [-step, 0, step]:
            for dy in [-step, 0, step]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (point[0] + dx, point[1] + dy)
                if self._is_valid(neighbor, boundaries, margin):
                    neighbors.append(neighbor)
        return neighbors

    def _has_line_of_sight(self, p1, p2, boundaries, resolution, margin):
        """Checks if the line segment is valid (respects margin)."""
        dist = self._distance(p1, p2)
        if dist < resolution:
            return True
        
        num_checks = int(dist / resolution)
        if num_checks == 0: return True
        
        dx = (p2[0] - p1[0]) / num_checks
        dy = (p2[1] - p1[1]) / num_checks

        for i in range(1, num_checks):
            intermediate_point = (p1[0] + i * dx, p1[1] + i * dy)
            if not self._is_valid(intermediate_point, boundaries, margin):
                return False
        return True

    # New helper to find shortest distance from a point to the whole boundary
    def _distance_to_boundary(self, point, boundaries):
        min_dist = float('inf')
        for i in range(len(boundaries)):
            p1 = boundaries[i]
            p2 = boundaries[(i + 1) % len(boundaries)]
            dist = self._point_to_segment_distance(point, p1, p2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    # New helper for point-to-line-segment distance calculation
    def _point_to_segment_distance(self, p, v, w):
        """Calculates shortest distance from point p to line segment vw."""
        l2 = self._distance(v, w)**2
        if l2 == 0.0:
            return self._distance(p, v)
        # Project p onto the line vw, but clamp t to [0,1] to stay on the segment
        t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        return self._distance(p, projection)
        
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