from common.data_models import Gear, Point, GearSet
import math

class GearFactory:
    def __init__(self, module: float = 1.0):
        self.module = module

    def create_gear_from_diameter(self, gear_id: str, center: tuple[float, float], desired_diameter: float) -> Gear:
        """
        Creates a gear by approximating a desired diameter.
        
        Args:
            desired_diameter: The target pitch diameter for the gear.
        
        Returns:
            A Gear object with an integer number of teeth and a diameter
            that is a close approximation of the desired diameter.
        """
        if desired_diameter <= 0:
            raise ValueError("Diameter must be positive.")

        # 1. Calculate the ideal, non-integer number of teeth
        ideal_teeth = desired_diameter / self.module
        
        # 2. Round to the nearest whole number for a valid tooth count
        actual_teeth = round(ideal_teeth)
        
        # Ensure the tooth count is within a valid range
        actual_teeth = max(8, min(200, actual_teeth))
        
        # 3. Create the gear using the valid, integer tooth count
        return self.create_gear(gear_id, center, actual_teeth)

    def create_gear(self, gear_id: str, center: tuple[float, float], num_teeth: int | list[int]):
        """
        Creates a gear set. Accepts a single int for a simple gear or a
        list of ints for a compound gear.
        """
        # Ensure we are always working with a list for consistent processing.
        teeth_list = [num_teeth] if isinstance(num_teeth, int) else num_teeth

        # Validate each tooth count in the list
        for teeth_count in teeth_list:
            if not (8 <= teeth_count <= 200):
                raise ValueError(
                    f"Tooth count {teeth_count} is not within the valid range of 8 to 200."
                )
            
        # Convert the center tuple into a Point object before creating the GearSet.
        center_point = Point(x=center[0], y=center[1])
        
        # Pass the Point object to the constructor
        return GearSet(id=gear_id, center=center_point, teeth_count=teeth_list, module=self.module)

    def get_meshing_distance(self, num_teeth1: int, num_teeth2: int) -> float:
        """Calculates the required center-to-center distance for two gears to mesh."""
        radius1 = (self.module * num_teeth1) / 2
        radius2 = (self.module * num_teeth2) / 2
        return radius1 + radius2

