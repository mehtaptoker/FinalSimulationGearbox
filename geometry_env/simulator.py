import math
import sys
import os
from typing import List, Tuple, Union
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from physics_validator.validator import PhysicsValidator
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.data_models import Point, GearSet
from gear_generator.factory import GearFactory


class GearTrainSimulator:
    def __init__(self, path: Union[List[List[float]], List[Point]], 
                 input_shaft: Union[Tuple[float, float], Point],
                 output_shaft: Union[Tuple[float, float], Point], 
                 boundaries: Union[List[List[float]], List[Point]],
                 gear_factory: GearFactory, clearance_margin: float = 1.0):

        # Normalize all geometry inputs to Point objects
        self.path = self._normalize_path(path)
        self.input_shaft = self._normalize_point(input_shaft)
        self.output_shaft = self._normalize_point(output_shaft)
        self.boundaries = self._normalize_boundaries(boundaries)

        # Factories / params
        self.gear_factory = gear_factory
        self.clearance_margin = max(0.2, clearance_margin * 0.5)  # Use 50% of specified clearance

        # State
        self.gears: List[GearSet] = []
        self.last_gear: GearSet = None
        self.input_gear: GearSet = None
        self.output_gear: GearSet = None
        self.distance_on_path: float = 0.0
        self.current_path_index = 1

    def _normalize_point(self, p) -> Point:
        """Convert any point format to Point object."""
        if isinstance(p, Point):
            return p
        if isinstance(p, dict) and 'x' in p and 'y' in p:
            return Point(x=float(p['x']), y=float(p['y']))
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            return Point(x=float(p[0]), y=float(p[1]))
        if hasattr(p, 'x') and hasattr(p, 'y'):
            return Point(x=float(p.x), y=float(p.y))
        raise ValueError(f"Cannot convert {type(p)} to Point: {p}")

    def _normalize_path(self, path) -> List[Point]:
        """Convert path to list of Point objects."""
        if not path:
            return []
        
        normalized = []
        for p in path:
            try:
                normalized.append(self._normalize_point(p))
            except Exception as e:
                print(f"Warning: Could not normalize path point {p}: {e}")
                continue
        return normalized

    def _normalize_boundaries(self, boundaries) -> List[Point]:
        """Convert boundaries to list of Point objects."""
        if not boundaries:
            return []
        
        normalized = []
        for b in boundaries:
            try:
                normalized.append(self._normalize_point(b))
            except Exception as e:
                print(f"Warning: Could not normalize boundary point {b}: {e}")
                continue
        return normalized

    # ------------------ Public API ------------------

    def reset(self, initial_gear_teeth: int = None):
        self.gears = []

        max_input_radius = self._distance_to_boundary(self.input_shaft, self.boundaries) - self.clearance_margin
        max_output_radius = self._distance_to_boundary(self.output_shaft, self.boundaries) - self.clearance_margin

        if max_input_radius <= 0 or max_output_radius <= 0:
            return self._get_state(), 0.0, True, {"error": "Input/Output shaft too close to boundary."}

        self.input_gear = self.gear_factory.create_gear_from_diameter(
            gear_id='gear_input',
            center=(self.input_shaft.x, self.input_shaft.y),
            desired_diameter=max_input_radius * 2.0
        )
        self.output_gear = self.gear_factory.create_gear_from_diameter(
            gear_id='gear_output',
            center=(self.output_shaft.x, self.output_shaft.y),
            desired_diameter=max_output_radius * 2.0
        )

        self.gears = [self.input_gear, self.output_gear]
        self.last_gear = self.input_gear
        self.distance_on_path = self._get_path_distance_of_point(self.input_shaft)
        self._prev_s = self.distance_on_path

        return self._get_state(), 0.0, False, {}

    def step(self, action: tuple):
        """
        Enhanced step function with better geometry validation.
        """
        driven_teeth, driving_teeth = action

        # Check for minimum viable gear sizes
        if driven_teeth < 8 or driving_teeth < 8:
            return self._get_state(), -5.0, False, {"error": "Gear teeth count too small (minimum 8)"}

        # --- 1. Always-tangent placement ---
        try:
            s_guess = self.distance_on_path + self.gear_factory.get_meshing_distance(
                self.last_gear.teeth_count[-1], driven_teeth
            )
            
            placement_result = self._place_tangent_to_prev(
                driven_teeth=driven_teeth,
                driving_teeth=driving_teeth,
                s_guess=s_guess
            )
        
            # Handle None return safely
            if placement_result is None or placement_result[0] is None:
                placed_gear, s_star, snap_error = None, None, None
            else:
                placed_gear, s_star, snap_error = placement_result
                
        except Exception as e:
            print(f"Warning: Placement calculation failed: {e}")
            placed_gear, s_star, snap_error = None, None, None

        # --- Invalid Placement Handling ---
        if placed_gear is None:
            try:
                alternative_result = self._try_alternative_placement(driven_teeth, driving_teeth)
                if alternative_result is None or alternative_result[0] is None:
                    alternative_gear, alt_s, alt_error = None, None, None
                else:
                    alternative_gear, alt_s, alt_error = alternative_result
            except Exception as e:
                print(f"Warning: Alternative placement failed: {e}")
                alternative_gear, alt_s, alt_error = None, None, None
            
            if alternative_gear is None:
                return self._get_state(), -10.0, False, {"error": "Invalid placement - too close to boundary or collision"}
            else:
                placed_gear, s_star, snap_error = alternative_gear, alt_s, alt_error
            driven_teeth, driving_teeth = action

            # Check for minimum viable gear sizes
            if driven_teeth < 8 or driving_teeth < 8:
                return self._get_state(), -5.0, False, {"error": "Gear teeth count too small (minimum 8)"}

        # --- 1. Always-tangent placement ---
        s_guess = self.distance_on_path + self.gear_factory.get_meshing_distance(
            self.last_gear.teeth_count[-1], driven_teeth
        )
        
        placed_gear, s_star, snap_error = self._place_tangent_to_prev(
            driven_teeth=driven_teeth,
            driving_teeth=driving_teeth,
            s_guess=s_guess
        )

        # --- Invalid Placement Handling ---
        if placed_gear is None:
            # Try alternative placement strategies before giving up
            alternative_gear, alt_s, alt_error = self._try_alternative_placement(driven_teeth, driving_teeth)
            
            if alternative_gear is None:
                return self._get_state(), -10.0, False, {"error": "Invalid placement - too close to boundary or collision"}
            else:
                placed_gear, s_star, snap_error = alternative_gear, alt_s, alt_error

        # --- Accept Valid Placement ---
        self.distance_on_path = s_star
        self.gears.insert(-1, placed_gear)
        self.last_gear = placed_gear

        # --- Reward for Progress ---
        progress = self.distance_on_path - self._prev_s
        reward = 0.1 * max(0.0, progress)
        self._prev_s = self.distance_on_path
        info = {}
        done = False

        # --- Success Condition 1: Direct Mesh ---
        if self.has_output_meshed(tol=0.5):
            reward += 100.0
            done = True
            info["success"] = "Gear train successfully meshed with output."
            return self._get_state(), reward, done, info

        # --- Success Condition 2: Snap to Output ---
        path_end_threshold = 35.0
        if (self._path_total_length() - self.distance_on_path) < path_end_threshold:
            snapped_gear, final_s, snap_err = self._snap_to_output()
            if snapped_gear is not None:
                self.gears[-2] = snapped_gear
                self.last_gear = snapped_gear
                self.distance_on_path = final_s
                reward += 100.0
                done = True
                info["success"] = f"Snapped to output (err={snap_err:.3f})"
                return self._get_state(), reward, done, info

        # --- End of Path Handling ---
        if self.distance_on_path >= self._path_total_length() - 1.0:
            done = True
            info["error"] = "Reached end of path without meshing."

        return self._get_state(), reward, done, info

    def _try_alternative_placement(self, driven_teeth: int, driving_teeth: int):
        """Try alternative placement strategies when normal placement fails."""
        
        # Strategy 1: Try smaller gear sizes
        for size_reduction in [2, 4, 6]:
            alt_driven = max(8, driven_teeth - size_reduction)
            alt_driving = max(8, driving_teeth - size_reduction)
            
            s_guess = self.distance_on_path + self.gear_factory.get_meshing_distance(
                self.last_gear.teeth_count[-1], alt_driven
            )
            
            placed_gear, s_star, error = self._place_tangent_to_prev(alt_driven, alt_driving, s_guess)
            if placed_gear is not None:
                return placed_gear, s_star, error
        
        # Strategy 2: Try advancing further along path
        for advance_distance in [2.0, 4.0, 6.0]:
            s_guess = self.distance_on_path + advance_distance
            placed_gear, s_star, error = self._place_tangent_to_prev(driven_teeth, driving_teeth, s_guess)
            if placed_gear is not None:
                return placed_gear, s_star, error
        
        return None, None, None

    # ------------------ State & helpers ------------------

    def _get_state(self):
        if not self.last_gear: 
            return None
        return {
            "last_gear_center_x": self.last_gear.center.x,
            "last_gear_center_y": self.last_gear.center.y,
            "last_gear_teeth": self.last_gear.teeth_count[-1],
            "last_gear_radius": self.last_gear.driving_radius,
            "target_gear_center_x": self.output_gear.center.x,
            "target_gear_center_y": self.output_gear.center.y,
            "distance_to_target": self._distance(self.last_gear.center, self.output_shaft),
        }

    def has_output_meshed(self, backlash: float = 0.0, tol: float = 0.2) -> bool:
        if not self.last_gear or not self.output_gear: 
            return False
        d = self._distance(self.last_gear.center, self.output_gear.center)
        required_dist = self.last_gear.driving_radius + self.output_gear.driven_radius - backlash
        return abs(required_dist - d) <= tol

    def _valid_center_with_margin(self, new_gear_set: GearSet, debug=False) -> bool:
        """Enhanced validation with better boundary checking."""
        max_r_new = max(new_gear_set.radii) if new_gear_set.radii else 0
        
        # Enhanced boundary checking
        distance_to_boundary = self._distance_to_boundary(new_gear_set.center, self.boundaries)
        required_boundary_distance = max_r_new + self.clearance_margin
        
        if distance_to_boundary < required_boundary_distance:
            if debug: 
                print(f"DEBUG: Boundary violation. Distance: {distance_to_boundary:.2f}, Required: {required_boundary_distance:.2f}")
            return False

        # Collision checking
        for g in self.gears:
            if g.id == new_gear_set.id or (self.last_gear and g.id == self.last_gear.id):
                continue

            margin = self.clearance_margin
            max_r_g = max(g.radii) if g.radii else 0
            required_separation = max_r_g + max_r_new + margin
            d = self._distance(g.center, new_gear_set.center)

            if d < required_separation - 1e-9:
                if debug: 
                    print(f"DEBUG: Collision with gear '{g.id}'. Distance: {d:.2f}, Required: {required_separation:.2f}")
                return False
        return True

    def _is_valid_placement(self, point, margin) -> bool:
        return self._distance_to_boundary(point, self.boundaries) >= margin

    def _find_point_on_path(self, target_distance: float):
        """Find point on path at given distance with robust error handling."""
        if not self.path or len(self.path) < 2:
            return None
            
        cumulative_dist = 0.0
        for i in range(len(self.path) - 1):
            p1, p2 = self.path[i], self.path[i+1]
            segment_len = self._distance(p1, p2)
            
            if cumulative_dist + segment_len >= target_distance:
                if segment_len < 1e-9: 
                    return Point(x=p1.x, y=p1.y)
                ratio = (target_distance - cumulative_dist) / segment_len
                ratio = max(0.0, min(1.0, ratio))  # Clamp ratio
                return Point(
                    x=p1.x + ratio * (p2.x - p1.x), 
                    y=p1.y + ratio * (p2.y - p1.y)
                )
            cumulative_dist += segment_len
        
        # Return last point if target_distance exceeds path length
        return self.path[-1] if self.path else None

    def _get_path_distance_of_point(self, point: Point) -> float:
        """Get distance along path to given point with robust handling."""
        if not self.path:
            return 0.0
            
        cumulative_dist, min_proj_dist, total_path_dist_to_proj = 0.0, float('inf'), 0.0
        
        for i in range(len(self.path) - 1):
            p1, p2 = self.path[i], self.path[i+1]
            segment_len = self._distance(p1, p2)
            
            if segment_len < 1e-9:
                cumulative_dist += segment_len
                continue
                
            l2 = segment_len ** 2
            t = max(0, min(1, ((point.x - p1.x) * (p2.x - p1.x) + (point.y - p1.y) * (p2.y - p1.y)) / l2))
            projection = Point(x=p1.x + t * (p2.x - p1.x), y=p1.y + t * (p2.y - p1.y))
            dist_to_proj = self._distance(point, projection)
            
            if dist_to_proj < min_proj_dist:
                min_proj_dist = dist_to_proj
                total_path_dist_to_proj = cumulative_dist + self._distance(p1, projection)
            cumulative_dist += segment_len
            
        return total_path_dist_to_proj

    def _distance(self, p1, p2):
        """Robust distance calculation between points."""
        try:
            if hasattr(p1, 'x') and hasattr(p1, 'y'):
                x1, y1 = float(p1.x), float(p1.y)
            else:
                x1, y1 = float(p1[0]), float(p1[1])
                
            if hasattr(p2, 'x') and hasattr(p2, 'y'):
                x2, y2 = float(p2.x), float(p2.y)
            else:
                x2, y2 = float(p2[0]), float(p2[1])
                
            return math.hypot(x1 - x2, y1 - y2)
        except Exception as e:
            print(f"Warning: Distance calculation failed for {p1}, {p2}: {e}")
            return float('inf')

    def _point_to_segment_distance(self, p, v, w):
        """Calculate distance from point to line segment."""
        try:
            # Normalize all inputs to Point objects
            if not isinstance(p, Point):
                p = self._normalize_point(p)
            if not isinstance(v, Point):
                v = self._normalize_point(v)
            if not isinstance(w, Point):
                w = self._normalize_point(w)
                
            l2 = self._distance(v, w) ** 2
            if l2 == 0.0: 
                return self._distance(p, v)
                
            t = max(0, min(1, ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2))
            proj = Point(x=v.x + t * (w.x - v.x), y=v.y + t * (w.y - v.y))
            return self._distance(p, proj)
        except Exception as e:
            print(f"Warning: Point-to-segment distance calculation failed: {e}")
            return float('inf')

    def _distance_to_boundary(self, point, boundaries):
        """Calculate minimum distance from point to boundary."""
        if not boundaries:
            return float('inf')
            
        min_dist = float('inf')
        for i in range(len(boundaries)):
            try:
                dist = self._point_to_segment_distance(
                    point, 
                    boundaries[i], 
                    boundaries[(i + 1) % len(boundaries)]
                )
                min_dist = min(min_dist, dist)
            except Exception as e:
                print(f"Warning: Boundary distance calculation failed: {e}")
                continue
        return min_dist

    def _path_total_length(self) -> float:
        """Calculate total path length with error handling."""
        if not self.path or len(self.path) < 2:
            return 0.0
        return sum(self._distance(self.path[i], self.path[i+1]) for i in range(len(self.path) - 1))

    def _distance_to_point_at(self, s: float, pt: Point) -> float:
        p = self._find_point_on_path(s)
        return self._distance(p, pt) if p else 1e9

    def _snap_along_path_to_distance(self, s_guess: float, target_point, required_dist: float,
                                    search_half_window: float = 25.0, tol: float = 0.15, iters: int = 30):
        """Snap positioning along path to achieve required distance."""
        s_min = max(0.0, s_guess - search_half_window)
        s_max = min(self._path_total_length(), s_guess + search_half_window)
        
        f = lambda s: self._distance_to_point_at(s, target_point) - required_dist
        fa, fb = f(s_min), f(s_max)

        if fa * fb > 0:
            s_best, f_best = (s_min, fa) if abs(fa) < abs(fb) else (s_max, fb)
            f_guess = f(s_guess)
            if abs(f_guess) < abs(f_best): 
                s_best, f_best = s_guess, f_guess
            return (s_best, self._find_point_on_path(s_best), f_best) if abs(f_best) <= tol else (None, None, None)

        a, b = s_min, s_max
        for _ in range(iters):
            m = (a + b) / 2
            if m == a or m == b: 
                break
            fm = f(m)
            if abs(fm) <= tol: 
                return m, self._find_point_on_path(m), fm
            if fa * fm < 0: 
                b = m
            else: 
                a, fa = m, fm
        return None, None, None

    def _place_tangent_to_prev(self, driven_teeth: int, driving_teeth: int, s_guess: float):
        """Place gear tangent to previous gear with enhanced validation."""
        try:
            temp_gear = self.gear_factory.create_gear(
                gear_id=f'gear_temp', 
                center=(0, 0),
                num_teeth=[driven_teeth, driving_teeth] if driven_teeth != driving_teeth else [driven_teeth]
            )
            req_dist = self.last_gear.driving_radius + temp_gear.driven_radius

            snap = self._snap_along_path_to_distance(
                s_guess=s_guess, 
                target_point=self.last_gear.center, 
                required_dist=req_dist
            )
            
            if snap[0] is None: 
                return None, None, None

            s_star, p_star, err = snap
            
            if p_star is None:
                return None, None, None
                
            final_gear = self.gear_factory.create_gear(
                gear_id=f'gear_{len(self.gears)-1}', 
                center=(p_star.x, p_star.y),
                num_teeth=temp_gear.teeth_count
            )

            return (final_gear, s_star, err) if self._valid_center_with_margin(final_gear) else (None, None, None)
            
        except Exception as e:
            print(f"Warning: Gear placement failed: {e}")
            return None, None, None

    def _snap_to_output(self):
        """Enhanced snap to output with better error handling."""
        if not self.last_gear: 
            return None, None, None

        try:
            req_dist = self.last_gear.driving_radius + self.output_gear.driven_radius
            snap = self._snap_along_path_to_distance(
                s_guess=self.distance_on_path,
                target_point=self.output_gear.center,
                required_dist=req_dist,
                search_half_window=20.0,
                tol=0.2
            )
            
            if snap[0] is None: 
                return None, None, None

            s_star, p_star, err = snap
            
            if p_star is None:
                return None, None, None
                
            snapped_gear = self.gear_factory.create_gear(
                gear_id=self.last_gear.id,
                center=(p_star.x, p_star.y),
                num_teeth=self.last_gear.teeth_count
            )

            return (snapped_gear, s_star, err) if self._valid_center_with_margin(snapped_gear) else (None, None, None)
    
        
        except Exception as e:
            print(f"Warning: Snap to output failed: {e}")
    
            return None, None, None