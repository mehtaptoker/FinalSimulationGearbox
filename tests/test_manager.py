import unittest
import json
import os
from geometry_env.manager import EnvironmentManager
from common.data_models import Point, Boundary

class TestEnvironmentManager(unittest.TestCase):
    def setUp(self):
        # Create test data directory if it doesn't exist
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create valid test JSON
        self.valid_json_path = os.path.join(self.test_data_dir, 'valid.json')
        with open(self.valid_json_path, 'w') as f:
            json.dump({
                "obstacles": [
                    [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}],
                    [{"x": 20, "y": 20}, {"x": 30, "y": 20}, {"x": 25, "y": 30}]
                ]
            }, f)
        
        # Create empty obstacles JSON
        self.empty_obstacles_path = os.path.join(self.test_data_dir, 'empty_obstacles.json')
        with open(self.empty_obstacles_path, 'w') as f:
            json.dump({"obstacles": []}, f)
        
        # Create malformed JSON
        self.malformed_path = os.path.join(self.test_data_dir, 'malformed.json')
        with open(self.malformed_path, 'w') as f:
            f.write('{"obstacles": [ [{"x": 0}]}')  # Missing closing brackets
            
        # Create invalid structure JSON
        self.invalid_structure_path = os.path.join(self.test_data_dir, 'invalid_structure.json')
        with open(self.invalid_structure_path, 'w') as f:
            json.dump({"invalid_key": []}, f)
            
        # Create invalid point format JSON
        self.invalid_point_path = os.path.join(self.test_data_dir, 'invalid_point.json')
        with open(self.invalid_point_path, 'w') as f:
            json.dump({
                "obstacles": [
                    [{"x": 0}, {"y": 10}, {"x": 10, "y": 10}]  # First point missing y
                ]
            }, f)
            
        # Create obstacle with too few points
        self.few_points_path = os.path.join(self.test_data_dir, 'few_points.json')
        with open(self.few_points_path, 'w') as f:
            json.dump({
                "obstacles": [
                    [{"x": 0, "y": 0}, {"x": 10, "y": 0}]  # Only 2 points
                ]
            }, f)

    def test_load_valid_json(self):
        """Test loading a valid JSON file with obstacles"""
        obstacles = EnvironmentManager.load_obstacles_from_json(self.valid_json_path)
        self.assertEqual(len(obstacles), 2)
        self.assertIsInstance(obstacles[0], Boundary)
        self.assertEqual(len(obstacles[0].vertices), 3)
        self.assertEqual(obstacles[0].vertices[0], Point(0, 0))
        self.assertEqual(obstacles[0].vertices[1], Point(10, 0))
        self.assertEqual(obstacles[0].vertices[2], Point(10, 10))
        
        # Verify shaft locations are correctly set
        self.assertEqual(EnvironmentManager.INPUT_SHAFT, Point(0.0, 0.0))
        self.assertEqual(EnvironmentManager.OUTPUT_SHAFT, Point(10.0, 0.0))

    def test_load_empty_obstacles(self):
        """Test loading JSON with empty obstacles list"""
        obstacles = EnvironmentManager.load_obstacles_from_json(self.empty_obstacles_path)
        self.assertEqual(len(obstacles), 0)

    def test_load_malformed_json(self):
        """Test loading a malformed JSON file"""
        with self.assertRaises(ValueError) as context:
            EnvironmentManager.load_obstacles_from_json(self.malformed_path)
        self.assertIn("Invalid JSON", str(context.exception))

    def test_load_invalid_structure(self):
        """Test loading JSON with invalid structure (missing obstacles key)"""
        with self.assertRaises(ValueError) as context:
            EnvironmentManager.load_obstacles_from_json(self.invalid_structure_path)
        self.assertIn("missing 'obstacles' key", str(context.exception))

    def test_load_invalid_point_format(self):
        """Test loading JSON with invalid point format"""
        with self.assertRaises(ValueError) as context:
            EnvironmentManager.load_obstacles_from_json(self.invalid_point_path)
        self.assertIn("Invalid point format", str(context.exception))

    def test_load_too_few_points(self):
        """Test loading obstacle with too few points"""
        with self.assertRaises(ValueError) as context:
            EnvironmentManager.load_obstacles_from_json(self.few_points_path)
        self.assertIn("at least 3 points", str(context.exception))

    def test_file_not_found(self):
        """Test loading non-existent file"""
        with self.assertRaises(FileNotFoundError):
            EnvironmentManager.load_obstacles_from_json("non_existent.json")

if __name__ == '__main__':
    unittest.main()
