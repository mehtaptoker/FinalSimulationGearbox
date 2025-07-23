import unittest
import os
import tempfile
from visualization.renderer import Renderer
from common.data_models import SystemDefinition, Boundary, Point, Gear

class TestRenderer(unittest.TestCase):
    def setUp(self):
        self.renderer = Renderer()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "output.png")
        
        # Create sample boundary
        self.boundary = Boundary(
            vertices=(
                Point(-40, -40),
                Point(40, -40),
                Point(40, 40),
                Point(-40, 40)
            )
        )
        
        # Create sample gears
        self.gears = [
            Gear(position=Point(0, 0), radius=10, teeth=20),
            Gear(position=Point(20, 20), radius=8, teeth=16),
            Gear(position=Point(-15, -15), radius=6, teeth=12)
        ]
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_render_system_with_boundary_and_gears(self):
        """Test rendering a system with boundary and gears"""
        system = SystemDefinition(
            name="Test System",
            boundary=self.boundary,
            gears=self.gears
        )
        
        # Should run without errors
        self.renderer.render_system(system, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
        self.assertGreater(os.path.getsize(self.output_path), 0)
        
    def test_render_system_with_boundary_only(self):
        """Test rendering a system with boundary but no gears"""
        system = SystemDefinition(
            name="Empty System",
            boundary=self.boundary,
            gears=[]
        )
        
        self.renderer.render_system(system, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
        self.assertGreater(os.path.getsize(self.output_path), 0)
        
    def test_render_system_with_gears_only(self):
        """Test rendering a system with gears but no boundary"""
        system = SystemDefinition(
            name="No Boundary System",
            boundary=None,
            gears=self.gears
        )
        
        self.renderer.render_system(system, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
        self.assertGreater(os.path.getsize(self.output_path), 0)
        
    def test_render_system_empty(self):
        """Test rendering a completely empty system"""
        system = SystemDefinition(
            name="Empty System",
            boundary=None,
            gears=[]
        )
        
        self.renderer.render_system(system, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
        self.assertGreater(os.path.getsize(self.output_path), 0)

if __name__ == "__main__":
    unittest.main()
