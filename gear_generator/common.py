from dataclasses import dataclass

@dataclass
class Gear:
    id: str
    center: tuple[float, float]  # (x, y) coordinates
    num_teeth: int
    module: float = 1.0  # Default module value
    pressure_angle: float = 20.0  # Default pressure angle in degrees
    z_layer: int = 0  # Default z-layer for 3D positioning
