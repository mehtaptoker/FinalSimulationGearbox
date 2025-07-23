import unittest
from pathfinding.finder import Pathfinder
import os

class TestPathfinder(unittest.TestCase):
    def setUp(self):
        self.finder = Pathfinder()
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        
    def test_simple_path(self):
        """Test pathfinding with no obstacles"""
        file_path = os.path.join(self.test_data_dir, 'simple_path.json')
        path = self.finder.find_path(file_path)
        
        # Should have at least start and end points
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (5, 5))
        
        # Path should be monotonically increasing in both axes
        prev = path[0]
        for point in path[1:]:
            self.assertGreaterEqual(point[0], prev[0])
            self.assertGreaterEqual(point[1], prev[1])
            prev = point

    def test_obstructed_path(self):
        """Test pathfinding with obstacles"""
        file_path = os.path.join(self.test_data_dir, 'obstructed_path.json')
        path = self.finder.find_path(file_path)
        
        # Should have at least start and end points
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (6, 6))
        
        # Verify path avoids the obstacle (2,2) to (4,4)
        for point in path:
            self.assertFalse(
                2 <= point[0] <= 4 and 2 <= point[1] <= 4,
                f"Path point {point} is inside obstacle"
            )

    def test_impossible_path(self):
        """Test pathfinding when no path exists"""
        file_path = os.path.join(self.test_data_dir, 'impossible_path.json')
        with self.assertRaises(ValueError) as context:
            self.finder.find_path(file_path)
        self.assertEqual(str(context.exception), "No valid path found")

if __name__ == '__main__':
    unittest.main()
