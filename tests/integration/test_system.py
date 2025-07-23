import os
import json
import pytest
from pathlib import Path
from common.data_models import SystemDefinition

@pytest.mark.integration
def test_end_to_end_system():
    """Test the full system workflow from input to output generation"""
    input_name = "Example1"
    
    # Run the main workflow
    os.system(f"python main.py --input_name {input_name}")
    
    # Verify outputs
    output_dir = Path(f"outputs/{input_name}")
    system_json = output_dir / "system.json"
    system_png = output_dir / "system.png"
    
    assert system_json.exists(), "System JSON output missing"
    assert system_png.exists(), "System visualization missing"
    
    # Validate JSON structure
    with open(system_json, 'r') as f:
        system_data = json.load(f)
    
    # Basic validation of SystemDefinition structure
    assert "input_name" in system_data
    assert "constraints" in system_data
    assert "gear_layout" in system_data
    assert system_data["input_name"] == input_name
    
    # Validate gear layout
    gear_layout = system_data["gear_layout"]
    assert "gears" in gear_layout
    assert "paths" in gear_layout
    assert len(gear_layout["gears"]) > 0, "No gears generated"
    
    # Test with invalid input
    with pytest.raises(SystemExit):
        os.system("python main.py --input_name InvalidExample")
