import json
import math
from typing import List, Tuple, Dict, Any, Set

class Pathfinder:
    def __init__(self):
        self.closed_set: Set[Tuple[float, float]] = set()

    def find_path(self, processed_data_path: str) -> List[Tuple[float, float]]:
        """
        Find a path from input shaft to output shaft using A* algorithm
        
        Args:
            processed_data_path: Path to JSON file containing:
                - boundaries: List of obstacle polygons
                - input_shaft: Starting point (x, y)
                - output_shaft: Target point (x, y)
        
        Returns:
            List of points representing the path
        """
        # Load and parse JSON data
        with open(processed_data_path, 'r') as f:
            data = json.load(f)
        
        # Extract boundaries - stored as a single polygon with multiple points
        raw_boundary = data.get('boundaries', [])
        boundaries = [raw_boundary] if raw_boundary else []
        
        input_shaft = (float(data['input_shaft']['x']), float(data['input_shaft']['y']))
        output_shaft = (float(data['output_shaft']['x']), float(data['output_shaft']['y']))
        
        # A* implementation
        open_set: Set[Tuple[float, float]] = {input_shaft}
        came_from: Dict[Tuple[float, float], Tuple[float, float]] = {}
        g_score: Dict[Tuple[float, float], float] = {input_shaft: 0}
        f_score: Dict[Tuple[float, float], float] = {input_shaft: self._heuristic(input_shaft, output_shaft)}
        self.closed_set = set()
        
        print(f"🚀 Starting pathfinding from {input_shaft} to {output_shaft}")
        print(f"Workspace boundary: {boundaries[0] if boundaries else 'None'}")
        print(f"Obstacles: {boundaries[1:]}")
        
        while open_set:
            current = min(open_set, key=lambda p: f_score.get(p, float('inf')))
            print(f"🔍 Current node: {current}, F-score: {f_score.get(current, float('inf'))}")
            
            if current == output_shaft:
                print("🎯 Reached target node")
                return self._reconstruct_path(came_from, current)
            
            open_set.remove(current)
            self.closed_set.add(current)
            
            neighbors = self._get_neighbors(current, boundaries)
            print(f"  ➕ Found {len(neighbors)} neighbors: {neighbors}")
            
            for neighbor in neighbors:
                if neighbor in self.closed_set:
                    continue
                    
                # Use higher cost for diagonal moves (sqrt(2) instead of 1)
                move_cost = 1.0 if current[0] == neighbor[0] or current[1] == neighbor[1] else math.sqrt(2)
                tentative_g = g_score[current] + move_cost
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, output_shaft)
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        
        raise ValueError("No valid path found")

    def _heuristic(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Euclidean distance heuristic"""
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Actual distance between two points"""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return math.sqrt(dx*dx + dy*dy)

    def _get_neighbors(self, point: Tuple[float, float], boundaries: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
        """Get valid neighboring points (avoiding boundaries)"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
        neighbors = []
        for dx, dy in directions:
            neighbor = (point[0] + dx, point[1] + dy)
            
            # Check if point is inside any boundary
            inside_boundary = False
        # The first boundary is the workspace (valid area)
        if boundaries:
            workspace = boundaries[0]
            float_workspace = [(float(pt[0]), float(pt[1])) for pt in workspace]
            
            # Point must be inside workspace to be valid
            if not self._point_in_polygon(neighbor, float_workspace):
                print(f"    ❌ Neighbor {neighbor} is outside workspace")
                return neighbors
                
        # Check if point is inside any obstacle (boundaries after the first)
        for obstacle in boundaries[1:]:
            float_obstacle = [(float(pt[0]), float(pt[1])) for pt in obstacle]
            if self._point_in_polygon(neighbor, float_obstacle):
                print(f"    ❌ Neighbor {neighbor} is inside obstacle {obstacle}")
                return neighbors
                
        # If we get here, the point is valid
        print(f"    ✅ Neighbor {neighbor} is valid")
        neighbors.append(neighbor)
        return neighbors

    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Determine if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            # Check if point is on the vertex
            if (x, y) == (p1x, p1y) or (x, y) == (p2x, p2y):
                return True
                
            # Check if point is on horizontal edge
            if p1y == p2y and y == p1y and min(p1x, p2x) <= x <= max(p1x, p2x):
                return True
                
            # Check if point is on vertical edge
            if p1x == p2x and x == p1x and min(p1y, p2y) <= y <= max(p1y, p2y):
                return True
                
            # Check for intersection
            if (p1y > y) != (p2y > y):
                xinters = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
                if x < xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _reconstruct_path(self, came_from: Dict[Tuple[float, float], Tuple[float, float]], 
                         current: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
