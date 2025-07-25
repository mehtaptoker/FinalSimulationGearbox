import math
from typing import List, Tuple
from common.data_models import Point


# from common.data_models import Gear

class GearTrainSimulator:
    def __init__(self, path: List[tuple], input_shaft: Tuple[float, float],
                 output_shaft: Tuple[float, float], boundaries: List[List[float]],
                 gear_factory, clearance_margin: float = 1.0):
        
        self.path = [Point(x=p[0], y=p[1]) for p in path]
        self.input_shaft = Point(x=input_shaft[0], y=input_shaft[1])
        self.output_shaft = Point(x=output_shaft[0], y=output_shaft[1])
        self.boundaries = [Point(x=p[0], y=p[1]) for p in boundaries]
        self.gear_factory = gear_factory
        self.clearance_margin = clearance_margin

        self.gears = []
        self.last_gear = None
        self.distance_on_path = 0.0 # NEW: Tracks progress along the path

    def reset(self, initial_gear_teeth: int = 20):
        self.gears = []
        
        first_gear = self.gear_factory.create_gear(
            gear_id='gear_0',
            center=(self.input_shaft.x, self.input_shaft.y),
            num_teeth=initial_gear_teeth
        )
        
        total_margin = first_gear.driven_radius + self.clearance_margin
        if not self._is_valid_placement(first_gear.center, total_margin):
            return None, 0, True, {"error": "Initial gear invalid"}

        self.gears.append(first_gear)
        self.last_gear = first_gear
        
        # Initialize distance by finding where the input shaft lies on the path
        self.distance_on_path = self._get_path_distance_of_point(self.input_shaft)
        
        return self._get_state(), 0, False, {}

    def step(self, action: tuple):
        driven_teeth, driving_teeth = action
        new_teeth_counts = [driven_teeth, driving_teeth] if driven_teeth != driving_teeth else [driven_teeth]
        
        meshing_dist = self.gear_factory.get_meshing_distance(
            self.last_gear.teeth_counts[-1],
            driven_teeth
        )
        
        # NEW: Calculate next position by adding to current distance
        self.distance_on_path += meshing_dist
        next_center = self._find_point_on_path(self.distance_on_path)

        if next_center is None:
            return self._get_state(), -100.0, True, {"error": "End of path reached"}

        new_gear_set = self.gear_factory.create_gear(
            gear_id=f'gear_{len(self.gears)}',
            center=(next_center.x, next_center.y),
            num_teeth=new_teeth_counts
        )
       
        max_radius = max(new_gear_set.radii)
        total_margin = max_radius + self.clearance_margin
        if not self._is_valid_placement(new_gear_set.center, total_margin):
            return self._get_state(), -100.0, True, {"error": "Invalid placement, too close to boundary"}

        self.gears.append(new_gear_set)
        self.last_gear = new_gear_set
        
        dist_to_target = self._distance(self.last_gear.center, self.output_shaft)
        if dist_to_target <= self.last_gear.driving_radius:
            return self._get_state(), 100.0, True, {"success": "Output shaft reached"}

        return self._get_state(), -1.0, False, {}

    def _find_point_on_path(self, target_distance: float):
        """Finds a point on the path at a specific distance from the start of the path."""
        cumulative_dist = 0.0
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            segment_len = self._distance(p1, p2)
            
            if cumulative_dist + segment_len >= target_distance:
                dist_into_segment = target_distance - cumulative_dist
                ratio = dist_into_segment / segment_len
                target_x = p1.x + ratio * (p2.x - p1.x)
                target_y = p1.y + ratio * (p2.y - p1.y)
                return Point(x=target_x, y=target_y)
                
            cumulative_dist += segment_len
        return None

    def _get_path_distance_of_point(self, point: Point) -> float:
        """Finds the cumulative distance along the path to the closest point on the path to the given point."""
        cumulative_dist = 0.0
        min_proj_dist = float('inf')
        total_path_dist_to_proj = 0.0

        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            
            # Find projection of the point onto the current segment
            l2 = self._distance(p1, p2)**2
            if l2 == 0: continue
            t = max(0, min(1, ((point.x - p1.x) * (p2.x - p1.x) + (point.y - p1.y) * (p2.y - p1.y)) / l2))
            projection = Point(x=p1.x + t * (p2.x - p1.x), y=p1.y + t * (p2.y - p1.y))
            
            dist_to_proj = self._distance(point, projection)
            if dist_to_proj < min_proj_dist:
                min_proj_dist = dist_to_proj
                total_path_dist_to_proj = cumulative_dist + self._distance(p1, projection)
                
            cumulative_dist += self._distance(p1, p2)
        return total_path_dist_to_proj

    def _get_state(self):
        """Compiles the current state of the simulation into a dictionary."""
        if not self.last_gear:
            return None
            
        # The state should reflect the 'driving' part of the last gear set.
        return {
            "last_gear_center_x": self.last_gear.center.x,
            "last_gear_center_y": self.last_gear.center.y,
            "last_gear_teeth": self.last_gear.teeth_counts[-1],
            "last_gear_radius": self.last_gear.driving_radius,
            "distance_to_target": self._distance(self.last_gear.center, self.output_shaft)
        }

    def _is_valid_placement(self, point, margin):
        """A point is valid if it's inside and respects the margin."""
        # This function assumes Pathfinder's helpers are available or re-implemented here.
        # For simplicity, let's assume a placeholder check.
        # In a real implementation, you would pass the Pathfinder or its methods in.
        # To make this self-contained, we'll just check the margin for now.
        if self._distance_to_boundary(point, self.boundaries) < margin:
            return False
        return True # Assumes it's already inside

    ### HELPER METHODS (Can be inherited or passed from Pathfinder) ###
    def _distance(self, p1, p2):
        """
        Calculates Euclidean distance between two points.
        Handles both Point objects and tuples.
        """
        x1 = p1.x if hasattr(p1, 'x') else p1[0]
        y1 = p1.y if hasattr(p1, 'y') else p1[1]
        x2 = p2.x if hasattr(p2, 'x') else p2[0]
        y2 = p2.y if hasattr(p2, 'y') else p2[1]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def _point_to_segment_distance(self, p, v, w):
        """
        Calculates the shortest distance from Point p to the line segment vw.
        All inputs are expected to be Point objects.
        """
        if not isinstance(p, Point):
            p = Point(x=p[0], y=p[1])
        if not isinstance(v, Point):
            v = Point(x=v[0], y=v[1])
        if not isinstance(w, Point):
            w = Point(x=w[0], y=w[1])
        l2 = self._distance(v, w)**2
        if l2 == 0.0:
            return self._distance(p, v)

        # Project p onto the line vw, clamping t to the segment [0,1]
        t = max(0, min(1, ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2))
        
        # Calculate the projection point's coordinates
        projection_point = (v.x + t * (w.x - v.x), v.y + t * (w.y - v.y))
        
        return self._distance(p, projection_point)

    def _distance_to_boundary(self, point, boundaries):
        min_dist = float('inf')
        for i in range(len(boundaries)):
            dist = self._point_to_segment_distance(point, boundaries[i], boundaries[(i + 1) % len(boundaries)])
            if dist < min_dist:
                min_dist = dist
        return min_dist
