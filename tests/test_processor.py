import unittest
import os
import json
import cv2
import numpy as np
from preprocessing.processor import Processor

class TestProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # Create test image with red and green shafts
        cls.img_path = os.path.join(cls.test_data_dir, 'test_image.png')
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 10, (0, 0, 255), -1)  # Red input shaft
        cv2.circle(img, (150, 150), 10, (0, 255, 0), -1)  # Green output shaft
        cv2.rectangle(img, (20, 20), (180, 180), (255, 255, 255), 2)  # Boundary
        cv2.imwrite(cls.img_path, img)
        
        # Create test constraints JSON
        cls.constraints_path = os.path.join(cls.test_data_dir, 'constraints.json')
        constraints = {
            "input_speed": 100,
            "output_torque": 50,
            "material": "steel"
        }
        with open(cls.constraints_path, 'w') as f:
            json.dump(constraints, f)
        
        cls.output_path = os.path.join(cls.test_data_dir, 'processed.json')

    def test_json_parsing(self):
        """Test loading of constraints JSON"""
        constraints = Processor.process_input(
            self.img_path, 
            self.constraints_path, 
            self.output_path
        )
        self.assertEqual(constraints['input_speed'], 100)
        self.assertEqual(constraints['output_torque'], 50)
        self.assertEqual(constraints['material'], 'steel')

    def test_shaft_detection(self):
        """Test detection of input/output shafts"""
        Processor.process_input(self.img_path, self.constraints_path, self.output_path)
        
        with open(self.output_path, 'r') as f:
            data = json.load(f)
            
        # Verify shaft positions
        self.assertAlmostEqual(data['input_shaft']['x'], 50, delta=2)
        self.assertAlmostEqual(data['input_shaft']['y'], 50, delta=2)
        self.assertAlmostEqual(data['output_shaft']['x'], 150, delta=2)
        self.assertAlmostEqual(data['output_shaft']['y'], 150, delta=2)

    def test_boundary_detection(self):
        """Test boundary contour approximation"""
        Processor.process_input(self.img_path, self.constraints_path, self.output_path)
        
        with open(self.output_path, 'r') as f:
            data = json.load(f)
            
        # Verify boundary points
        points = np.array(data['boundaries'])
        self.assertTrue(len(points) > 0, "No boundary points detected")
        
        # Check that boundary covers the expected area
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        self.assertAlmostEqual(min_x, 20, delta=5)
        self.assertAlmostEqual(min_y, 20, delta=5)
        self.assertAlmostEqual(max_x, 180, delta=5)
        self.assertAlmostEqual(max_y, 180, delta=5)

    def test_invalid_image_path(self):
        """Test handling of missing image file"""
        with self.assertRaises(FileNotFoundError):
            Processor.process_input(
                'invalid_path.png',
                self.constraints_path,
                self.output_path
            )

    def test_invalid_json_path(self):
        """Test handling of missing constraints file"""
        with self.assertRaises(FileNotFoundError):
            Processor.process_input(
                self.img_path,
                'invalid_path.json',
                self.output_path
            )

    def test_output_creation(self):
        """Test output file creation"""
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
            
        Processor.process_input(self.img_path, self.constraints_path, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

    @classmethod
    def tearDownClass(cls):
        # Clean up test files
        for path in [cls.img_path, cls.constraints_path, cls.output_path]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == '__main__':
    unittest.main()
