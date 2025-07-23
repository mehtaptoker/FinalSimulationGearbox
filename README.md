# GearRL: Reinforcement Learning for Gear System Design

GearRL is an integrated system that generates optimized gear layouts from input images and constraints using reinforcement learning.

## System Overview
- **Preprocessing**: Converts input images/constraints to normalized geometry
- **Pathfinding**: Identifies optimal gear placement paths
- **Gear Generation**: Creates gear systems using RL optimization
- **Physics Validation**: Ensures mechanical feasibility
- **Visualization**: Renders final designs

## Installation
```bash
pip install -r requirements.txt
```

## Usage
To generate a gear system from input files:
```bash
python main.py --input_name Example1
```

### Input Requirements
1. `data/{input_name}.png` - Input image (PNG format)
2. `data/{input_name}_constraints.json` - Design constraints

### Outputs
Results are saved to:
- `outputs/{input_name}/system.json` - Final system definition
- `outputs/{input_name}/system.png` - Visualization

## Testing
Run integration tests:
```bash
pytest tests/integration
