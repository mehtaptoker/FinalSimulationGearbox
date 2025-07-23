import unittest
from common.data_models import Point
from gear_generator.factory import GearFactory

class TestGearFactory(unittest.TestCase):
    def setUp(self):
        self.factory = GearFactory()
        
    def test_create_valid_gear(self):
        """Test creating gears with valid tooth counts"""
        test_cases = [
            (8, 0.75),    # Minimum teeth
            (20, 0.75),   # Boundary
            (21, 1.0),    # Boundary +1
            (40, 1.0),    # Boundary
            (41, 1.25),   # Boundary +1
            (60, 1.25),   # Boundary
            (61, 1.5),    # Boundary +1
            (80, 1.5),    # Boundary
            (81, 2.0),    # Boundary +1
            (200, 2.0)    # Maximum teeth
        ]
        
        for teeth, expected_module in test_cases:
            with self.subTest(teeth=teeth):
                center = Point(0, 0)
                gear = self.factory.create_gear(center, teeth)
                
                self.assertEqual(gear.center, center)
                self.assertEqual(gear.teeth, teeth)
                self.assertEqual(gear.module, expected_module)
                self.assertEqual(gear.diameter, teeth * expected_module)

    def test_invalid_teeth_count(self):
        """Test creating gears with invalid tooth counts"""
        test_cases = [7, 0, -5, 201, 300]
        
        for teeth in test_cases:
            with self.subTest(teeth=teeth), \
                 self.assertRaises(ValueError) as context:
                center = Point(0, 0)
                self.factory.create_gear(center, teeth)
                
            if teeth < 8:
                self.assertEqual(str(context.exception), "Gear must have at least 8 teeth")
            else:
                self.assertEqual(str(context.exception), "Gear cannot have more than 200 teeth")

if __name__ == '__main__':
    unittest.main()
