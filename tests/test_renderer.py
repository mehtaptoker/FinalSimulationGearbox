import pytest
import tempfile
import os
from visualization.renderer import Renderer
from common.data_models import SystemDefinition, Boundary, Point, Constraints

@pytest.fixture
def sample_system():
    """Create a sample SystemDefinition for testing."""
    boundary = Boundary(points=[
        Point(x=-50, y=-50),
        Point(x=50, y=-50),
        Point(x=50, y=50),
        Point(x=-50, y=50)
    ])
    return SystemDefinition(
        boundary=boundary,
        input_shaft=Point(x=-30, y=0),
        output_shaft=Point(x=30, y=0),
        constraints=Constraints(
            torque_ratio="1:1",
            mass_space_ratio=0.5,
            boundary_margin=5.0,
            min_gear_size=10,
            max_gear_size=50
        )
    )

def test_render_system(sample_system):
    """Test that render_system creates an output file."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        Renderer.render_system(sample_system, output_path)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

def test_render_empty_boundary():
    """Test rendering with empty boundary points."""
    system = SystemDefinition(
        boundary=Boundary(points=[]),
        input_shaft=Point(x=0, y=0),
        output_shaft=Point(x=0, y=0),
        constraints=Constraints(
            torque_ratio="1:1",
            mass_space_ratio=0.5,
            boundary_margin=5.0,
            min_gear_size=10,
            max_gear_size=50
        )
    )
    
    with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
        with pytest.raises(ValueError):
            Renderer.render_system(system, tmp.name)
