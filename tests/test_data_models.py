import sys
from pathlib import Path
import unittest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.data_models import Constraints, Point, Boundary, Gear, GearLayout, SystemDefinition, ValidationReport

class TestConstraints(unittest.TestCase):
    def test_creation(self):
        constraints = Constraints(
            torque_ratio="1:2",
            mass_space_ratio=0.7,
            boundary_margin=10.0,
            min_gear_size=15,
            max_gear_size=50
        )
        self.assertEqual(constraints.torque_ratio, "1:2")
        self.assertEqual(constraints.mass_space_ratio, 0.7)
        
    def test_serialization(self):
        constraints = Constraints("1:3", 0.8, 5.0, 10, 40)
        json_data = constraints.to_json()
        self.assertEqual(json_data["torque_ratio"], "1:3")
        self.assertEqual(json_data["min_gear_size"], 10)
        
    def test_deserialization(self):
        json_data = {
            "torque_ratio": "2:1",
            "mass_space_ratio": 0.6,
            "boundary_margin": 15.0,
            "min_gear_size": 20,
            "max_gear_size": 60
        }
        constraints = Constraints.from_json(json_data)
        self.assertEqual(constraints.mass_space_ratio, 0.6)
        self.assertEqual(constraints.max_gear_size, 60)

class TestPoint(unittest.TestCase):
    def test_creation(self):
        point = Point(10.5, 20.3)
        self.assertEqual(point.x, 10.5)
        self.assertEqual(point.y, 20.3)
        
    def test_serialization(self):
        point = Point(30.1, 40.2)
        json_data = point.to_json()
        self.assertEqual(json_data["x"], 30.1)
        self.assertEqual(json_data["y"], 40.2)
        
    def test_deserialization(self):
        json_data = {"x": 100.0, "y": 200.0}
        point = Point.from_json(json_data)
        self.assertEqual(point.x, 100.0)
        self.assertEqual(point.y, 200.0)

class TestBoundary(unittest.TestCase):
    def test_creation(self):
        points = [Point(0,0), Point(100,0), Point(100,100), Point(0,100)]
        boundary = Boundary(points)
        self.assertEqual(len(boundary.points), 4)
        self.assertEqual(boundary.points[2].x, 100)
        
    def test_serialization(self):
        points = [Point(1,2), Point(3,4)]
        boundary = Boundary(points)
        json_data = boundary.to_json()
        self.assertEqual(len(json_data), 2)
        self.assertEqual(json_data[1]["x"], 3)
        
    def test_deserialization(self):
        json_data = [{"x": 5, "y": 6}, {"x": 7, "y": 8}]
        boundary = Boundary.from_json(json_data)
        self.assertEqual(boundary.points[0].y, 6)
        self.assertEqual(boundary.points[1].x, 7)

class TestGear(unittest.TestCase):
    def test_creation(self):
        gear = Gear(center=Point(50,50), teeth=20, module=2.0)
        self.assertEqual(gear.teeth, 20)
        self.assertEqual(gear.module, 2.0)
        
    def test_diameter_property(self):
        gear = Gear(center=Point(0,0), teeth=30, module=1.5)
        self.assertAlmostEqual(gear.diameter, 45.0)
        
    def test_serialization(self):
        gear = Gear(center=Point(10,20), teeth=25, module=1.8)
        json_data = gear.to_json()
        self.assertEqual(json_data["teeth"], 25)
        self.assertEqual(json_data["center"]["x"], 10)
        
    def test_deserialization(self):
        json_data = {"center": {"x": 30, "y": 40}, "teeth": 40, "module": 2.5}
        gear = Gear.from_json(json_data)
        self.assertEqual(gear.center.y, 40)
        self.assertAlmostEqual(gear.diameter, 100.0)

class TestGearLayout(unittest.TestCase):
    def test_creation(self):
        gears = [
            Gear(Point(0,0), 20, 1.5),
            Gear(Point(50,50), 30, 1.5)
        ]
        layout = GearLayout(gears)
        self.assertEqual(len(layout.gears), 2)
        self.assertEqual(layout.gears[1].teeth, 30)
        
    def test_serialization(self):
        gears = [Gear(Point(10,10), 15, 2.0)]
        layout = GearLayout(gears)
        json_data = layout.to_json()
        self.assertEqual(len(json_data), 1)
        self.assertEqual(json_data[0]["module"], 2.0)
        
    def test_deserialization(self):
        json_data = [{"center": {"x": 20, "y": 30}, "teeth": 25, "module": 1.8}]
        layout = GearLayout.from_json(json_data)
        self.assertEqual(layout.gears[0].center.x, 20)
        self.assertAlmostEqual(layout.gears[0].diameter, 45.0)

class TestSystemDefinition(unittest.TestCase):
    def test_creation(self):
        boundary = Boundary([Point(0,0), Point(100,0), Point(100,100), Point(0,100)])
        input_shaft = Point(10,10)
        output_shaft = Point(90,90)
        constraints = Constraints("1:1", 0.5, 5.0, 10, 30)
        
        system = SystemDefinition(
            boundary=boundary,
            input_shaft=input_shaft,
            output_shaft=output_shaft,
            constraints=constraints
        )
        
        self.assertEqual(system.input_shaft.x, 10)
        self.assertEqual(system.constraints.min_gear_size, 10)
        
    def test_serialization(self):
        boundary = Boundary([Point(1,2)])
        input_shaft = Point(3,4)
        output_shaft = Point(5,6)
        constraints = Constraints("2:1", 0.6, 10.0, 15, 40)
        
        system = SystemDefinition(boundary, input_shaft, output_shaft, constraints)
        json_data = system.to_json()
        
        self.assertEqual(json_data["input_shaft"]["x"], 3)
        self.assertEqual(json_data["constraints"]["torque_ratio"], "2:1")
        self.assertEqual(len(json_data["boundary_poly"]), 1)
        
    def test_deserialization(self):
        json_data = {
            "boundary_poly": [{"x": 0, "y": 0}, {"x": 200, "y": 0}, {"x": 200, "y": 200}],
            "input_shaft": {"x": 50, "y": 50},
            "output_shaft": {"x": 150, "y": 150},
            "constraints": {
                "torque_ratio": "3:1",
                "mass_space_ratio": 0.75,
                "boundary_margin": 15.0,
                "min_gear_size": 20,
                "max_gear_size": 60
            }
        }
        
        system = SystemDefinition.from_json(json_data)
        self.assertEqual(len(system.boundary.points), 3)
        self.assertEqual(system.output_shaft.y, 150)
        self.assertEqual(system.constraints.max_gear_size, 60)

class TestValidationReport(unittest.TestCase):
    def test_creation(self):
        report = ValidationReport(is_valid=False, errors=["Gear overlap", "Invalid ratio"])
        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.errors), 2)
        
    def test_serialization(self):
        report = ValidationReport(is_valid=True)
        json_data = report.to_json()
        self.assertTrue(json_data["is_valid"])
        self.assertEqual(len(json_data["errors"]), 0)
        
    def test_with_errors(self):
        report = ValidationReport(is_valid=False, errors=["Constraint violation"])
        json_data = report.to_json()
        self.assertFalse(json_data["is_valid"])
        self.assertEqual(json_data["errors"][0], "Constraint violation")

if __name__ == "__main__":
    unittest.main()
