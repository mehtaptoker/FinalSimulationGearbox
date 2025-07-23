from common.data_models import Gear, Point
import math

class GearFactory:
    def __init__(self, base_module: float = 1.0):
        """
        Initialize the gear factory with a base module size
        
        Args:
            base_module: Standard size for gear teeth (default 1.0)
        """
        self.base_module = base_module

    def create_gear(self, center: Point, teeth: int) -> Gear:
        """
        Create a gear with specified center and number of teeth
        
        Args:
            center: Center point of the gear (x, y)
            teeth: Number of teeth on the gear
            
        Returns:
            Gear object with calculated properties
        """
        # Validate input
        if teeth < 8:
            raise ValueError("Gear must have at least 8 teeth")
        if teeth > 200:
            raise ValueError("Gear cannot have more than 200 teeth")
            
        # Calculate gear properties
        module = self._calculate_module(teeth)
        return Gear(center=center, teeth=teeth, module=module)

    def _calculate_module(self, teeth: int) -> float:
        """
        Calculate module based on number of teeth using standard gear formulas
        """
        # Standard module sizing based on tooth count
        if teeth <= 20:
            return self.base_module * 0.75
        elif teeth <= 40:
            return self.base_module
        elif teeth <= 60:
            return self.base_module * 1.25
        elif teeth <= 80:
            return self.base_module * 1.5
        else:
            return self.base_module * 2.0
