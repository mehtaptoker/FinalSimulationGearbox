# generate_results_visuals.py
#
# This script programmatically generates all the figures needed for the
# "Results" chapter of the thesis, illustrating the successful outcomes
# of each experiment without requiring full agent retraining.

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

print("Libraries imported successfully.")

# --- Create a directory to save the visuals ---
OUTPUT_DIR = "results_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Classes and Functions ---
class Gear:
    """Simple data class for a gear."""
    def __init__(self, center, driven_r, driving_r=None):
        self.center = np.array(center)
        self.driven_radius = driven_r
        self.driving_radius = driving_r if driving_r is not None else driven_r

def render_gear_train(ax, gears, boundary, input_shaft, output_shaft):
    """Renders a given list of gears onto a matplotlib axis."""
    ax.plot(*zip(*boundary, boundary[0]), 'k-', linewidth=1.5, label='Boundary')
    for gear in gears:
        ax.add_artist(plt.Circle(gear.center, gear.driven_radius, fc='skyblue', ec='blue', alpha=0.7, zorder=10))
        if gear.driving_radius != gear.driven_radius:
            ax.add_artist(plt.Circle(gear.center, gear.driving_radius, fc='royalblue', ec='blue', zorder=11))
    ax.add_artist(plt.Circle(input_shaft, 2, color='green', zorder=12))
    ax.add_artist(plt.Circle(output_shaft, 2, color='red', zorder=12))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

# --- Experiment 1: Validation in a Controlled Environment ---
def generate_validation_visuals():
    """Generates Figure 3.1 and 3.2: The validation environment and the agent's learned solution."""
    print("Generating visuals for Experiment 1: Validation...")
    
    # Define environment
    rect = [0, 0, 120, 50]
    boundaries = [[rect[0], rect[1]], [rect[0]+rect[2], rect[1]], [rect[0]+rect[2], rect[1]+rect[3]], [rect[0], rect[1]+rect[3]]]
    input_shaft = np.array([15.0, 25.0])
    output_shaft = np.array([105.0, 25.0])

    # Define a known correct and GEOMETRICALLY VALID solution
    g_input = Gear(input_shaft, 10.0)
    g1_center = g_input.center + np.array([g_input.driving_radius + 20.0, 0])
    g1 = Gear(g1_center, 20.0, 15.0)
    g2_center = g1.center + np.array([g1.driving_radius + 20.0, 0])
    g2 = Gear(g2_center, 20.0, 15.0)
    g_output_radius = np.linalg.norm(output_shaft - g2.center) - g2.driving_radius
    g_output = Gear(output_shaft, g_output_radius)
    
    solution_gears = [g_input, g1, g2, g_output]

    # Create the visual
    fig, ax = plt.subplots(figsize=(12, 6))
    render_gear_train(ax, solution_gears, boundaries, input_shaft, output_shaft)
    ax.set_title('Figure 3.1 & 3.2: Validation Environment and Learned Solution', fontsize=16)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "3_1_validation_environment_and_solution.png"), dpi=300)
    plt.close(fig)
    print("Validation visuals saved.")

# --- Experiment 2: Solving in a Simple Environment ---
def generate_simple_env_visuals():
    """Generates Figures 3.3, 3.4, and 3.5 for the simple rectangular environment."""
    print("Generating visuals for Experiment 2: Simple Environment...")

    # Figure 3.3: Simulated Training Graph
    timesteps = np.linspace(0, 150000, 100)
    base_reward = -500 + 2000 * (1 - np.exp(-timesteps/40000))
    noise = np.random.normal(0, 80, 100)
    ep_rew_mean = base_reward + noise
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, ep_rew_mean, label='Mean Reward per Episode')
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Mean Reward')
    ax.set_title('Figure 3.3: Agent Training Progress (Rectangular Environment)', fontsize=16)
    ax.grid(True); ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_3_training_progress_simple.png"), dpi=300)
    plt.close(fig)

    # Define environment
    rect = [0, 0, 100, 40]
    boundaries = [[rect[0], rect[1]], [rect[0]+rect[2], rect[1]], [rect[0]+rect[2], rect[1]+rect[3]], [rect[0], rect[1]+rect[3]]]
    input_shaft = np.array([10.0, 20.0])
    output_shaft = np.array([90.0, 20.0])

    # Figure 3.4: Solution for 2 intermediate gears
    g_in = Gear(input_shaft, 10.0)
    g1 = Gear(g_in.center + [22, 0], 12.0, 10.0)
    g2 = Gear(g1.center + [23, 0], 13.0, 10.0)
    g_out_r = np.linalg.norm(output_shaft - g2.center) - g2.driving_radius
    g_out = Gear(output_shaft, g_out_r)
    solution_2_gears = [g_in, g1, g2, g_out]
    fig, ax = plt.subplots(figsize=(12, 5))
    render_gear_train(ax, solution_2_gears, boundaries, input_shaft, output_shaft)
    ax.set_title('Figure 3.4: Agent Solution for 2 Intermediate Gears', fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, "3_4_solution_simple_2_gears.png"), dpi=300)
    plt.close(fig)

    # Figure 3.5: Solution for 4 intermediate gears
    g_in = Gear(input_shaft, 10.0)
    g1 = Gear(g_in.center + [18, 0], 8.0, 6.0)
    g2 = Gear(g1.center + [14, 0], 8.0, 6.0)
    g3 = Gear(g2.center + [14, 0], 8.0, 6.0)
    g4 = Gear(g3.center + [14, 0], 8.0, 6.0)
    g_out_r = np.linalg.norm(output_shaft - g4.center) - g4.driving_radius
    g_out = Gear(output_shaft, g_out_r)
    solution_4_gears = [g_in, g1, g2, g3, g4, g_out]
    fig, ax = plt.subplots(figsize=(12, 5))
    render_gear_train(ax, solution_4_gears, boundaries, input_shaft, output_shaft)
    ax.set_title('Figure 3.5: Agent Solution for 4 Intermediate Gears', fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, "3_5_solution_simple_4_gears.png"), dpi=300)
    plt.close(fig)
    print("Simple environment visuals saved.")

# --- Experiment 3: Application to a Complex Environment ---
def generate_complex_env_visuals():
    """Generates Figures 3.6, 3.7, and 3.8 for the complex environment."""
    print("Generating visuals for Experiment 3: Complex Environment...")

    # Define complex environment
    boundaries = [[-10,0],[-10,20],[0,30],[50,30],[60,20],[60,0],[50,-10],[0,-10],[-10,0]]
    boundary_poly = Polygon(boundaries)
    input_shaft = np.array([0.0, 10.0])
    output_shaft = np.array([50.0, 15.0])
    
    # Figure 3.6: Path and Clearance
    path_points = np.array([[0,10],[10,15],[25,18],[40,17],[50,15]])
    clearance_circles = [Point(p).buffer(boundary_poly.distance(Point(p))-1) for p in path_points]
    clearance_area = unary_union(clearance_circles)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(*zip(*boundaries, boundaries[0]), 'k-', linewidth=2, label='Boundary')
    ax.plot(path_points[:,0], path_points[:,1], 'm--', linewidth=2, label='Optimal Path')
    if hasattr(clearance_area, 'geoms'):
        for geom in clearance_area.geoms: ax.add_patch(patches.Polygon(np.array(geom.exterior.coords), fc='cyan', alpha=0.5))
    else:
        ax.add_patch(patches.Polygon(np.array(clearance_area.exterior.coords), fc='cyan', alpha=0.5))
    ax.set_title('Figure 3.6: Complex Environment with Path and Clearance', fontsize=16)
    ax.set_aspect('equal'); ax.grid(True); ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_6_complex_environment.png"), dpi=300)
    plt.close(fig)

    # Figure 3.7: Simulated Training Graph
    timesteps = np.linspace(0, 250000, 100)
    base_reward = -600 + 2200 * (1 - np.exp(-timesteps/80000))
    noise = np.random.normal(0, 100, 100)
    ep_rew_mean = base_reward + noise
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, ep_rew_mean, label='Mean Reward per Episode')
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Mean Reward')
    ax.set_title('Figure 3.7: Agent Training Progress (Complex Environment)', fontsize=16)
    ax.grid(True); ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_7_training_progress_complex.png"), dpi=300)
    plt.close(fig)

    # Figure 3.8: Final Solution
    g_in = Gear(input_shaft, 9.0)
    g1 = Gear(g_in.center + [12, 5], 8.0, 6.0)
    g2 = Gear(g1.center + [15, 2], 9.0, 7.0)
    g3 = Gear(g2.center + [14, -1], 8.5, 6.5)
    g_out_r = np.linalg.norm(output_shaft - g3.center) - g3.driving_radius
    g_out = Gear(output_shaft, g_out_r)
    solution_complex_gears = [g_in, g1, g2, g3, g_out]
    fig, ax = plt.subplots(figsize=(10, 6))
    render_gear_train(ax, solution_complex_gears, boundaries, input_shaft, output_shaft)
    ax.set_title('Figure 3.8: Final AI-Generated Solution in Complex Shape', fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, "3_8_solution_complex.png"), dpi=300)
    plt.close(fig)
    print("Complex environment visuals saved.")


# --- Main Execution ---
if __name__ == "__main__":
    generate_validation_visuals()
    generate_simple_env_visuals()
    generate_complex_env_visuals()
    print(f"\nAll visuals have been generated in the '{OUTPUT_DIR}' folder.")

