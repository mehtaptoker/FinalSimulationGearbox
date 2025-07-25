import math
from common.data_models import ValidationReport, Constraints
def compute_reward(
    report: ValidationReport, 
    constraints: Constraints, 
    target_torque: float,
    torque_weight: float,
    space_weight: float,
    weight_penalty_coef: float
) -> float:
    """
    Calculate reward based on validation report and constraints
    
    Args:
        report: Validation report from physics validator
        constraints: Design constraints
        target_torque: Desired torque ratio
        torque_weight: Weight for torque component
        space_weight: Weight for space usage component
        weight_penalty_coef: Weight penalty coefficient
        
    Returns:
        float: Calculated reward scalar
    """
    # Heavy penalty for invalid designs (collisions, etc)
    if not report.is_valid:
        return -10.0  # Significant penalty for invalid designs
    
    # Calculate torque reward (exponential decay for closeness to target)
    torque_diff = abs(report.torque_ratio - target_torque)
    torque_reward = math.exp(-torque_diff)  # [0,1] range
    
    # Calculate space usage reward (higher = better)
    space_reward = report.space_usage
    
    # Calculate weight penalty (lower mass = better)
    weight_penalty = report.total_mass * 0.01  # Scale mass penalty
    
    # Weighted sum of components
    reward = (
        torque_weight * torque_reward + 
        space_weight * space_reward - 
        weight_penalty_coef * weight_penalty
    )
    return reward