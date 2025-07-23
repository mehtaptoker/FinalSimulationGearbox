import unittest
import os
from pathfinding.finder import Pathfinder

class TestPathfinder(unittest.TestCase):
    def setUp(self):
        self.finder = Pathfinder()
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')

    def test_simple_path(self):
        """Test pathfinding with no obstacles"""
        path = self.finder.find_path(os.path.join(self.test_data_dir, 'simple_path.json'))
        self.assertIsNotNone(path, "Should find a path in simple scenario")
        self.assertGreater(len(path), 1, "Path should have multiple points")

    def test_obstructed_path(self):
        """Test pathfinding with obstacles"""
        path = self.finder.find_path(os.path.join(self.test_data_dir, 'obstructed_path.json'))
        self.assertIsNotNone(path, "Should find a path around obstacles")
        self.assertGreater(len(path), 1, "Path should have multiple points")

    def test_impossible_path(self):
        """Test pathfinding when no path exists"""
        path = self.finder.find_path(os.path.join(self.test_data_dir, 'impossible_path.json'))
        self.assertIsNone(path, "Should return None when no path exists")

if __name__ == '__main__':
    unittest.main()
