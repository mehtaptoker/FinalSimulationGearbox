from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
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

@dataclass
class Gear:
    center: Point
    teeth: int
    module: float
    
    @property
    def diameter(self) -> float:
        return self.teeth * self.module
    
    @classmethod
    def from_json(cls, json_data: Dict) -> "Gear":
        return cls(
            center=Point.from_json(json_data["center"]),
            teeth=json_data["teeth"],
            module=json_data["module"]
        )
    
    def to_json(self) -> Dict:
        return asdict(self)

@dataclass
class GearLayout:
    gears: List[Gear]
    
    @classmethod
    def from_json(cls, json_data: List[Dict]) -> "GearLayout":
        return cls(gears=[Gear.from_json(g) for g in json_data])
    
    def to_json(self) -> List[Dict]:
        return [g.to_json() for g in self.gears]

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
