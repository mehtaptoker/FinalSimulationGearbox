from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Gear:
    id: str
    center: tuple[float, float]  # (x, y) coordinates
    num_teeth: int
    module: float = 1.0  # Default module value
    pressure_angle: float = 20.0  # Default pressure angle in degrees
    z_layer: int = 0  # Default z-layer for 3D positioning

    # A calculated property for the pitch radius
    pitch_radius: float = field(init=False, repr=True)

    def __post_init__(self):
        """Calculate radius after the object is created."""
        self.pitch_radius = (self.module * self.num_teeth) / 2


@dataclass
class GearSet:
    """Represents one or more gears fixed to a single shaft."""
    id: str
    center: Tuple[float, float]
    
    # A list of teeth counts for gears on this shaft.
    # e.g., [20] is a simple gear.
    # e.g., [40, 15] is a compound gear.
    diameter_ref: List[int]
    
    module: float = 1.0
    
    # A list of corresponding pitch radii will be calculated.
    radii: List[float] = field(init=False, repr=True)

    def __post_init__(self):
        """Calculate properties after the object is created."""
        if not self.diameter_ref:
            raise ValueError("GearSet must have at least one gear.")
        self.radii = [(self.module * teeth) / 2 for teeth in self.diameter_ref]

    @property
    def driven_radius(self) -> float:
        """The radius of the gear that meshes with the *previous* gear set."""
        return self.radii[0]
        
    @property
    def driving_radius(self) -> float:
        """The radius of the gear that will drive the *next* gear set."""
        return self.radii[-1] # The last gear in the set

