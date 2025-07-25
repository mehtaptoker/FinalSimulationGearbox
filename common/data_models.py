from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
import json

@dataclass
class Constraints:
    torque_ratio: str
    mass_space_ratio: float
    boundary_margin: float
    min_gear_size: int
    max_gear_size: int

    @classmethod
    def from_json(cls, json_data: Dict) -> "Constraints":
        return cls(
            torque_ratio=json_data["torque_ratio"],
            mass_space_ratio=json_data["mass_space_ratio"],
            boundary_margin=json_data["boundary_margin"],
            min_gear_size=json_data["min_gear_size"],
            max_gear_size=json_data["max_gear_size"]
        )
    
    def to_json(self) -> Dict:
        return asdict(self)

@dataclass
class Point:
    x: float
    y: float

    @classmethod
    def from_json(cls, json_data: Dict) -> "Point":
        return cls(x=json_data["x"], y=json_data["y"])
    
    def to_json(self) -> Dict:
        return asdict(self)

@dataclass
class Boundary:
    points: List[Point]

    @classmethod
    def from_json(cls, json_data: List[Dict]) -> "Boundary":
        return cls(points=[Point.from_json(p) for p in json_data])
    
    def to_json(self) -> List[Dict]:
        return [p.to_json() for p in self.points]

# @dataclass
# class Gear:
#     center: Point
#     teeth: int
#     module: float
    
#     @property
#     def diameter(self) -> float:
#         return self.teeth * self.module
    
#     @classmethod
#     def from_json(cls, json_data: Dict) -> "Gear":
#         return cls(
#             center=Point.from_json(json_data["center"]),
#             teeth=json_data["teeth"],
#             module=json_data["module"]
#         )
    
#     def to_json(self) -> Dict:
#         return asdict(self)



@dataclass
class SystemDefinition:
    boundary: Boundary
    input_shaft: Point
    output_shaft: Point
    constraints: Constraints
    
    @classmethod
    def from_json(cls, json_data: Dict) -> "SystemDefinition":
        return cls(
            boundary=Boundary.from_json(json_data["boundary_poly"]),
            input_shaft=Point.from_json(json_data["input_shaft"]),
            output_shaft=Point.from_json(json_data["output_shaft"]),
            constraints=Constraints.from_json(json_data["constraints"])
        )
    
    def to_json(self) -> Dict:
        return {
            "boundary_poly": self.boundary.to_json(),
            "input_shaft": self.input_shaft.to_json(),
            "output_shaft": self.output_shaft.to_json(),
            "constraints": self.constraints.to_json()
        }

@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    
    def to_json(self) -> Dict:
        return asdict(self)


@dataclass
class Gear:
    """
    Represents a gear set on a single shaft. Can be a simple gear (one
    teeth count) or a compound gear (multiple teeth counts).
    """
    id: str
    center: Point
    teeth_count: List[int]
    module: float

    # The diameters list is calculated automatically after initialization
    diameters: List[float] = field(init=False, repr=True)

    def __post_init__(self):
        """Calculates derived properties after the object is created."""
        if not self.teeth_count:
            raise ValueError("Gear must have at least one teeth count.")
        self.diameters = [teeth * self.module for teeth in self.teeth_count]

    @property
    def driven_diameter(self) -> float:
        """Diameter of the gear that meshes with the *previous* gear in a train."""
        return self.diameters[0]

    @property
    def driving_diameter(self) -> float:
        """Diameter of the gear that drives the *next* gear in a train."""
        return self.diameters[-1]

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "Gear":
        """
        Creates a Gear instance from a dictionary.

        Handles both the new 'teeth_count' (list) and old 'teeth' (int) keys
        for backward compatibility.
        """
        teeth_input = json_data.get("teeth_count") or json_data["teeth"]
        
        # Ensure the final teeth_count is a list
        teeth_list = [teeth_input] if isinstance(teeth_input, int) else teeth_input

        return cls(
            id=json_data["id"],
            center=Point.from_json(json_data["center"]),
            teeth_count=teeth_list,
            module=json_data["module"]
        )

    def to_json(self) -> Dict[str, Any]:
        """Serializes the Gear object to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "center": self.center.to_json(),
            "teeth_count": self.teeth_count,
            "module": self.module
        }


@dataclass
class GearSet:
    """
    Represents a gear set on a single shaft. Can be a simple gear (one
    teeth count) or a compound gear (multiple teeth counts).
    Includes serialization methods.
    """
    id: str
    center: Point
    teeth_count: List[int]
    module: float

    # The diameters and radii lists are calculated automatically
    radii: List[float] = field(init=False, repr=True)
    diameters: List[float] = field(init=False, repr=False)

    def __post_init__(self):
        """Calculates derived properties after the object is created."""
        if not self.teeth_count:
            raise ValueError("GearSet must have at least one teeth count.")
        self.radii = [(teeth * self.module) / 2 for teeth in self.teeth_count]
        self.diameters = [r * 2 for r in self.radii]

    @property
    def driven_radius(self) -> float:
        """Radius of the gear that meshes with the previous gear in a train."""
        return self.radii[0]

    @property
    def driving_radius(self) -> float:
        """Radius of the gear that drives the next gear in a train."""
        return self.radii[-1]
    
    # --- ADD THESE METHODS ---

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "GearSet":
        """Creates a GearSet instance from a dictionary."""
        return cls(
            id=data["id"],
            center=Point.from_json(data["center"]),
            teeth_count=data["teeth_count"],
            module=data["module"]
        )

    def to_json(self) -> Dict[str, Any]:
        """Serializes the GearSet object to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "center": self.center.to_json(),
            "teeth_count": self.teeth_count,
            "module": self.module
        }



@dataclass
class GearLayout:
    gears: List[Gear]
    
    @classmethod
    def from_json(cls, json_data: List[Dict]) -> "GearLayout":
        return cls(gears=[Gear.from_json(g) for g in json_data])
    
    def to_json(self) -> List[Dict]:
        return [g.to_json() for g in self.gears]
