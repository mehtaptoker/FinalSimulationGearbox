import unittest
import math
from physics_validator.validator import PhysicsValidator
from common.data_models import GearLayout, SystemDefinition, Gear, Point, Constraints, Boundary, ValidationReport

class TestPhysicsValidator(unittest.TestCase):
    def setUp(self):
        # Common constraints
        self.constraints = Constraints(
            torque_ratio="2:1",
            mass_space_ratio=0.7,
            boundary_margin=0.1,
            min_gear_size=10,
            max_gear_size=50
        )
        
        # Boundary polygon (larger square to accommodate gears)
        self.boundary = Boundary(points=[
            Point(0, 0),
            Point(0, 10),
            Point(10, 10),
            Point(10, 0)
        ])
        
        # Input/output shafts
        self.input_shaft = Point(1, 1)
        self.output_shaft = Point(9, 9)
        
    def create_system(self):
        return SystemDefinition(
            boundary=self.boundary,
            input_shaft=self.input_shaft,
            output_shaft=self.output_shaft,
            constraints=self.constraints
        )
    
    def test_valid_layout(self):
        """Test a layout that should pass all validations"""
        layout = GearLayout(gears=[
            Gear(center=Point(1, 1), teeth=20, module=0.05),   # Input gear
            Gear(center=Point(9, 9), teeth=10, module=0.05),   # Output gear
            Gear(center=Point(4, 6), teeth=30, module=0.05)    # Intermediate
        ])
        
        system = self.create_system()
        report = PhysicsValidator.check_layout(layout, system)
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.errors), 0)
    
    def test_gear_collision(self):
        """Test overlapping gears"""
        layout = GearLayout(gears=[
            Gear(center=Point(0.3, 0.3), teeth=20, module=0.05),
            Gear(center=Point(0.31, 0.31), teeth=15, module=0.05)  # Too close
        ])
        
        system = self.create_system()
        report = PhysicsValidator.check_layout(layout, system)
        self.assertFalse(report.is_valid)
        self.assertIn("Gear collision", report.errors[0])
    
    def test_boundary_violation(self):
        """Test gear outside boundary"""
        layout = GearLayout(gears=[
            Gear(center=Point(11, 5), teeth=20, module=0.05)  # Outside boundary (x=11 > boundary max x=10)
        ])
        
        system = self.create_system()
        report = PhysicsValidator.check_layout(layout, system)
        self.assertFalse(report.is_valid)
        self.assertIn("outside boundary", report.errors[0])
    
    def test_boundary_margin_violation(self):
        """Test gear too close to boundary"""
        layout = GearLayout(gears=[
            # Diameter = 20*0.05 = 1.0 → radius=0.5
            # Center at (0.4, 5) → distance to left boundary (x=0) is 0.4
            # Required distance: 0.5 (radius) + 0.1 (margin) = 0.6
            Gear(center=Point(0.4, 5), teeth=20, module=0.05)
        ])
        
        system = self.create_system()
        report = PhysicsValidator.check_layout(layout, system)
        self.assertFalse(report.is_valid)
        self.assertIn("too close to boundary", report.errors[0])
    
    def test_gear_too_small(self):
        """Test gear below minimum size"""
        layout = GearLayout(gears=[
            Gear(center=Point(0.5, 0.5), teeth=8, module=0.05)  # Min size=10
        ])
        
        system = self.create_system()
        report = PhysicsValidator.check_layout(layout, system)
        self.assertFalse(report.is_valid)
        self.assertIn("too small", report.errors[0])
    
    def test_gear_too_large(self):
        """Test gear above maximum size"""
        layout = GearLayout(gears=[
            Gear(center=Point(5, 5), teeth=55, module=0.05)  # Max size=50
        ])
        
        system = self.create_system()
        report = PhysicsValidator.check_layout(layout, system)
        self.assertFalse(report.is_valid)
        self.assertIn("too large", report.errors[0])
    
    def test_torque_ratio_mismatch(self):
        """Test incorrect torque ratio"""
        layout = GearLayout(gears=[
            Gear(center=Point(1, 1), teeth=20, module=0.05),  # Input
            Gear(center=Point(9, 9), teeth=15, module=0.05)   # Output (ratio=1.33 vs target 2.0)
        ])
        
        system = self.create_system()
        report = PhysicsValidator.check_layout(layout, system)
        self.assertFalse(report.is_valid)
        self.assertIn("Torque ratio mismatch", report.errors[0])
    
    def test_free_torque_ratio(self):
        """Test free torque ratio constraint"""
        # Create constraints with free ratio
        constraints = Constraints(
            torque_ratio="free",
            mass_space_ratio=0.7,
            boundary_margin=0.1,
            min_gear_size=10,
            max_gear_size=50
        )
        
        layout = GearLayout(gears=[
            Gear(center=Point(1, 1), teeth=20, module=0.05),
            Gear(center=Point(9, 9), teeth=15, module=0.05)
        ])
        
        system = SystemDefinition(
            boundary=self.boundary,
            input_shaft=self.input_shaft,
            output_shaft=self.output_shaft,
            constraints=constraints
        )
        
        report = PhysicsValidator.check_layout(layout, system)
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.errors), 0)
    
    def test_single_gear(self):
        """Test layout with only one gear"""
        # Create constraints with free ratio since we can't calculate ratio with one gear
        constraints = Constraints(
            torque_ratio="free",
            mass_space_ratio=0.7,
            boundary_margin=0.1,
            min_gear_size=10,
            max_gear_size=50
        )
        
        layout = GearLayout(gears=[
            Gear(center=Point(5, 5), teeth=20, module=0.05)
        ])
        
        system = SystemDefinition(
            boundary=self.boundary,
            input_shaft=self.input_shaft,
            output_shaft=self.output_shaft,
            constraints=constraints
        )
        
        report = PhysicsValidator.check_layout(layout, system)
        self.assertTrue(report.is_valid)  # Should be valid with free ratio
        self.assertEqual(len(report.errors), 0)

if __name__ == '__main__':
    unittest.main()
