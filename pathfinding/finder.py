import json
import math
from heapq import heappop, heappush

class Pathfinder:
    def __init__(self):
        self.output_shaft = None  # Store output shaft for line-of-sight checks

    def find_path(self, processed_data_path):
        """
        Find path from input shaft to output shaft using A* algorithm
        
        Args:
            processed_data_path (str): Path to processed JSON file
            
        Returns:
            list: Path as list of [x, y] points in normalized space
        """
        # Load and parse JSON data
        with open(processed_data_path, 'r') as f:
            data = json.load(f)
            
        # Extract normalized space data (handle both real data and test data formats)
        if 'normalized_space' in data:
            norm_space = data['normalized_space']
            boundaries = norm_space['boundaries']
            input_shaft = (norm_space['input_shaft']['x'], norm_space['input_shaft']['y'])
            output_shaft = (norm_space['output_shaft']['x'], norm_space['output_shaft']['y'])
        else:
            boundaries = data['boundaries']
            input_shaft = (data['input_shaft'][0], data['input_shaft'][1])
            output_shaft = (data['output_shaft'][0], data['output_shaft'][1])
            
        # Store output shaft for line-of-sight checks
        self.output_shaft = output_shaft
        
        # Debug output
        print(f"Input shaft: {input_shaft}")
        print(f"Output shaft: {output_shaft}")
        print(f"Boundaries: {boundaries}")
        
        # A* implementation
        open_set = []
        closed_set = set()
        
        # Add start node
        heappush(open_set, (0, input_shaft))
        came_from = {}
        g_score = {input_shaft: 0}
        f_score = {input_shaft: self.heuristic(input_shaft, output_shaft)}
        
        # Debug: count iterations
        iterations = 0
        
        while open_set:
            iterations += 1
            _, current = heappop(open_set)
            
            if current == output_shaft:
                print(f"Path found after exploring {iterations} nodes")
                return self.reconstruct_path(came_from, current)
                
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current, boundaries):
                if neighbor in closed_set:
                    continue
                    
                tentative_g = g_score[current] + self.distance(current, neighbor)
                
                if neighbor not in [i[1] for i in open_set] or tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, output_shaft)
                    heappush(open_set, (f_score[neighbor], neighbor))
                    
        print(f"Explored {iterations} nodes, no path found")
        return None  # No path found

    def heuristic(self, a, b):
        """Euclidean distance heuristic"""
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        
    def distance(self, a, b):
        """Distance between two points"""
        return self.heuristic(a, b)
        
    def get_neighbors(self, point, boundaries, step=0.5):
        """Generate valid neighbor points within boundaries"""
        neighbors = []
        directions = [
            (step, 0), (-step, 0), (0, step), (0, -step),
            (step, step), (step, -step), (-step, step), (-step, -step)
        ]
        
        # Add larger diagonal steps for efficiency
        large_step = step * 2
        large_directions = [
            (large_step, large_step), (large_step, -large_step),
            (-large_step, large_step), (-large_step, -large_step)
        ]
        
        # Check direct line of sight to goal as a potential shortcut
        if self.has_line_of_sight(point, self.output_shaft, boundaries):
            neighbors.append(self.output_shaft)
            
        for dx, dy in directions + large_directions:
            neighbor = (point[0] + dx, point[1] + dy)
            if self.is_within_boundaries(neighbor, boundaries):
                neighbors.append(neighbor)
                
        return neighbors
        
    def has_line_of_sight(self, a, b, boundaries):
        """Check if there's a direct path between two points"""
        # Simple check - midpoint in boundaries (for convex shapes)
        mid = ((a[0] + b[0])/2, (a[1] + b[1])/2)
        return self.is_within_boundaries(mid, boundaries)
        
    def is_within_boundaries(self, point, boundaries):
        """Check if point is within polygon boundaries using ray casting"""
        if not boundaries:
            # If no boundaries defined, consider entire space valid
            return True
            
        x, y = point
        n = len(boundaries)
        inside = False
        
        # Get first boundary point
        p1 = boundaries[0]
        p1x, p1y = p1[0], p1[1]
        
        for i in range(1, n + 1):
            p2 = boundaries[i % n]
            p2x, p2y = p2[0], p2[1]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
        
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from start to current point"""
        path = [list(current)]
        while current in came_from:
            current = came_from[current]
            path.append(list(current))
        path.reverse()
        return path

    