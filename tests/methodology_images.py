# generate_methodology_visuals.py
#
# This script generates specific, high-quality visuals to explain the
# Reinforcement Learning methodology used in the thesis.

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon

print("Libraries imported successfully.")

# --- Create a directory to save the visuals ---
OUTPUT_DIR = "methodology_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Visual 1: The RL Environment Setup ---
def generate_environment_visual():
    """Generates a visual that explains the components of the RL environment."""
    print("Generating Visual 1: RL Environment Setup...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define the environment geometry
    rect = [0, 0, 100, 40]
    boundaries = [[rect[0], rect[1]], [rect[0]+rect[2], rect[1]], [rect[0]+rect[2], rect[1]+rect[3]], [rect[0], rect[1]+rect[3]]]
    input_shaft_pos = np.array([10.0, 20.0])
    output_shaft_pos = np.array([90.0, 20.0])
    
    # Plot the boundary and shafts
    ax.add_patch(patches.Polygon(boundaries, closed=True, fill=False, edgecolor='black', linewidth=2))
    ax.add_patch(plt.Circle(input_shaft_pos, 10.0, facecolor='skyblue', edgecolor='blue', alpha=0.7))
    ax.text(input_shaft_pos[0], input_shaft_pos[1], 'Input Gear\n(Fixed Start)', ha='center', va='center', fontsize=10)
    ax.add_patch(plt.Circle(output_shaft_pos, 2, color='red'))

    # Annotations to explain the components
    ax.annotate('Boundary (Polygon)', xy=(50, 40), xytext=(50, 45),
                arrowprops=dict(facecolor='black', shrink=0.05), ha='center', fontsize=12)

    ax.annotate('Output Shaft (Target)', xy=(90, 20), xytext=(75, 28),
                arrowprops=dict(facecolor='black', shrink=0.05), ha='left', fontsize=12)

    # State/Observation Vector Explanation
    state_text = "Agent's Observation (The State):\n" \
                 "- Vector to Target (x, y)\n" \
                 "- Current Gear Ratio\n" \
                 "- Number of Gears Placed"
    ax.text(0.98, 0.02, state_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_title('Methodology Visual 1: The Reinforcement Learning Environment', fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 50)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "methodology_1_environment.png"), dpi=300)
    plt.close(fig)
    print("Visual 1 saved.")

# --- Visual 2: The Action Space ---
def generate_action_space_visual():
    """Generates a visual explaining the agent's discrete action space."""
    print("Generating Visual 2: The Action Space...")
    fig, ax = plt.subplots(figsize=(12, 7))

    possible_actions = [5.0, 8.0, 10.0, 12.0, 15.0]
    
    # Previous gear for context
    last_gear_center = np.array([20, 20])
    last_gear_radius = 10.0
    ax.add_patch(plt.Circle(last_gear_center, last_gear_radius, facecolor='lightgray', edgecolor='black', alpha=0.8))
    ax.text(last_gear_center[0], last_gear_center[1], 'Last Gear', ha='center', va='center', fontsize=10)

    # Show the possible next gears
    for i, radius in enumerate(possible_actions):
        meshing_dist = last_gear_radius + radius
        center = last_gear_center + np.array([meshing_dist, 0])
        
        # Draw the gear
        ax.add_patch(plt.Circle(center, radius, facecolor='skyblue', edgecolor='blue', alpha=0.7, linestyle='--'))
        # Label the action
        ax.text(center[0], center[1] + radius + 2, f'Action {i}\n(r={radius})', ha='center', va='bottom', fontsize=10)

    ax.set_title('Methodology Visual 2: The Agent\'s Discrete Action Space', fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_xlim(10, 100)
    ax.set_ylim(0, 40)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "methodology_2_action_space.png"), dpi=300)
    plt.close(fig)
    print("Visual 2 saved.")

# --- Visual 3: The Reward Function Logic ---
def generate_reward_logic_visual():
    """Generates a flowchart explaining the reward function logic."""
    print("Generating Visual 3: Reward Function Logic...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Using text boxes and arrows to create a flowchart
    props = dict(boxstyle='round,pad=0.5', fc='wheat', ec='black', lw=1)
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', ec='black')

    # Boxes
    ax.text(0.5, 0.9, 'Agent takes an Action (places a gear)', ha='center', va='center', bbox=props, fontsize=12)
    ax.text(0.5, 0.7, 'Is the new gear out of bounds?', ha='center', va='center', bbox=props, fontsize=12)
    ax.text(0.15, 0.5, 'YES', ha='center', va='center', fontsize=10)
    ax.text(0.85, 0.5, 'NO', ha='center', va='center', fontsize=10)
    ax.text(0.15, 0.3, 'Reward = -200\n(Large Penalty)\nEpisode Ends', ha='center', va='center', bbox=dict(boxstyle='round', fc='salmon'), fontsize=12)
    
    ax.text(0.85, 0.55, 'Can the output shaft be connected?', ha='center', va='center', bbox=props, fontsize=12)
    ax.text(0.5, 0.4, 'NO', ha='center', va='center', fontsize=10)
    ax.text(0.5, 0.25, 'Reward = -1\n(Small Step Penalty)\nContinue Episode', ha='center', va='center', bbox=dict(boxstyle='round', fc='lightgray'), fontsize=12)

    ax.text(0.85, 0.4, 'YES', ha='center', va='center', fontsize=10)
    ax.text(0.85, 0.25, 'Is (Num Gears == Target) AND (Ratio ≈ Target)?', ha='center', va='center', bbox=props, fontsize=12)
    
    ax.text(0.6, 0.1, 'YES', ha='center', va='center', fontsize=10)
    ax.text(0.6, 0.0, 'Reward = +2000\n(Huge Bonus)\nEpisode Ends', ha='center', va='center', bbox=dict(boxstyle='round', fc='lightgreen'), fontsize=12)
    
    ax.text(1.1, 0.1, 'NO', ha='center', va='center', fontsize=10)
    ax.text(1.1, 0.0, 'Reward = -500\n(Penalty)\nEpisode Ends', ha='center', va='center', bbox=dict(boxstyle='round', fc='salmon'), fontsize=12)

    # Arrows
    ax.annotate('', xy=(0.5, 0.85), xytext=(0.5, 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(0.2, 0.65), xytext=(0.45, 0.7), arrowprops=arrow_props)
    ax.annotate('', xy=(0.8, 0.65), xytext=(0.55, 0.7), arrowprops=arrow_props)
    ax.annotate('', xy=(0.15, 0.45), xytext=(0.15, 0.35), arrowprops=arrow_props)
    ax.annotate('', xy=(0.85, 0.5), xytext=(0.85, 0.45), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.5), xytext=(0.8, 0.55), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.3), xytext=(0.5, 0.3), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.1', ec='black'))
    ax.annotate('', xy=(0.85, 0.2), xytext=(0.85, 0.3), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.2), xytext=(0.8, 0.25), arrowprops=arrow_props)
    ax.annotate('', xy=(1.05, 0.2), xytext=(0.9, 0.25), arrowprops=arrow_props)
    
    ax.set_title('Methodology Visual 3: The Reward Function Flowchart', fontsize=16)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(-0.1, 1.0)
    ax.axis('off')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "methodology_3_reward_logic.png"), dpi=300)
    plt.close(fig)
    print("Visual 3 saved.")

# --- Main Execution ---
if __name__ == "__main__":
    generate_environment_visual()
    generate_action_space_visual()
    generate_reward_logic_visual()
    print(f"\nAll visuals have been generated in the '{OUTPUT_DIR}' folder.")

################################
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

# --- Placeholder Data (Replace with your actual data) ---
# 1. Complex boundary coordinates
BOUNDARY_COORDS = [
    (0, 0), (10, 2), (12, 8), (8, 11), (3, 10), (1, 5), (0, 0)
]

# 2. Fixed shaft positions
INPUT_SHAFT_POS = (2, 7)
OUTPUT_SHAFT_POS = (10, 4)

# 3. Physical requirements
TARGET_RATIO = 2.0
NUM_GEARS = 3
# --- End of Placeholder Data ---

# Create the polygon object
boundary_polygon = Polygon(BOUNDARY_COORDS)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the boundary
x, y = boundary_polygon.exterior.xy
ax.plot(x, y, color='black', linewidth=2, label='Environment Boundary')
ax.fill(x, y, color='lightgray', alpha=0.5)

# Plot the shafts
ax.plot(INPUT_SHAFT_POS[0], INPUT_SHAFT_POS[1], 'ro', markersize=10, label='Input Shaft')
ax.plot(OUTPUT_SHAFT_POS[0], OUTPUT_SHAFT_POS[1], 'bo', markersize=10, label='Output Shaft')

# Add a text box for physical requirements
requirements_text = (
    f"Physical Requirements:\n"
    f"--------------------\n"
    f"- Target Gear Ratio: {TARGET_RATIO}\n"
    f"- Number of Gears: {NUM_GEARS}"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
ax.text(0.05, 0.95, requirements_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)


# Set plot properties
ax.set_title('Figure X: The Problem Definition', fontsize=14, weight='bold')
ax.set_xlabel('X Coordinate (mm)')
ax.set_ylabel('Y Coordinate (mm)')
ax.legend()
ax.set_aspect('equal', 'box')
ax.grid(True, linestyle='--', alpha=0.6)

# Save the figure
plt.tight_layout()
plt.savefig('figure_problem_definition.png', dpi=300)
plt.show()
########################################
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Define positions
agent_pos = (0.3, 0.5)
action_positions = [(0.7, 0.8), (0.7, 0.5), (0.7, 0.2)]
action_labels = ["Select Gear Radius: 5mm", "Select Gear Radius: 10mm", "Select Gear Radius: 15mm"]

# Draw the agent's current state
ax.text(agent_pos[0], agent_pos[1], "Agent at State S_t\n(Last Gear Placed)",
        ha='center', va='center', size=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", lw=1))

# Draw the actions and connecting arrows
for i, pos in enumerate(action_positions):
    ax.text(pos[0], pos[1], f"Action {i+1}:\n{action_labels[i]}",
            ha='center', va='center', size=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="black", lw=1))

    arrow = patches.FancyArrowPatch(
        (agent_pos[0] + 0.1, agent_pos[1]),  # Offset to start from the edge of the box
        (pos[0] - 0.1, pos[1]),        # Offset to end at the edge of the box
        connectionstyle="arc3,rad=.2",
        arrowstyle="->,head_width=15,head_length=15",
        color="gray",
        linewidth=2
    )
    ax.add_patch(arrow)

# Set plot properties
ax.set_title('Figure Y: The Discrete Action Space', fontsize=14, weight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off') # Hide axes for a diagrammatic look

# Save the figure
plt.tight_layout()
plt.savefig('figure_discrete_action_space.png', dpi=300)
plt.show()
####################
from graphviz import Digraph

# Create a new Digraph
dot = Digraph(comment='Reward Function Logic')
dot.attr(rankdir='TB', splines='ortho') # Top-to-Bottom layout, orthogonal lines
dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightgrey')
dot.attr('edge', fontsize='10')

# Define nodes
dot.node('start', 'Agent Takes Action\n(Places a gear)', shape='ellipse', fillcolor='lightblue')
dot.node('check_valid', 'Is Placement Valid?\n(Inside Boundary & No Overlap)', shape='diamond', fillcolor='wheat')
dot.node('penalty', 'Apply Large Negative Penalty\n R = -10', shape='box', fillcolor='salmon')
dot.node('calc_dist', 'Calculate Distance Reward\n(Closer to target is better)')
dot.node('check_goal', 'Is Final Goal Met?\n(Correct # of gears & ratio)', shape='diamond', fillcolor='wheat')
dot.node('reward_goal', 'Apply Large Goal Bonus\n R_goal = +50', fillcolor='lightgreen')
dot.node('calc_ratio', 'Calculate Ratio Reward\n(Closer to target is better)')
dot.node('sum_reward', 'Total Reward =\n R_dist + R_ratio + R_goal', shape='box', fillcolor='lightgreen')
dot.node('end', 'Return Total Reward', shape='ellipse', fillcolor='lightblue')

# Define edges (the logic flow)
dot.edge('start', 'check_valid')
dot.edge('check_valid', 'penalty', label=' No ')
dot.edge('check_valid', 'calc_dist', label=' Yes ')
dot.edge('penalty', 'end')
dot.edge('calc_dist', 'calc_ratio')
dot.edge('calc_ratio', 'check_goal')
dot.edge('check_goal', 'reward_goal', label=' Yes ')
dot.edge('check_goal', 'sum_reward', label=' No ')
dot.edge('reward_goal', 'sum_reward')
dot.edge('sum_reward', 'end')


# Set the title for the graph
dot.attr(label=r'\nFigure Z: The Reward Function Logic', fontsize='16', fontname='bold')

# Render and save the flowchart
# The 'render' method will save a '.gv' file and a '.gv.png' file
dot.render('figure_reward_function_logic', format='png', view=False, cleanup=True)

print("Flowchart 'figure_reward_function_logic.png' has been created.")###

import matplotlib.pyplot as plt
import numpy as np

# --- Placeholder Data (Replace with your actual data) ---
# Create some plausible sample data for demonstration
episodes = np.arange(0, 500, 10) # Logged every 10 episodes
# Simulate learning: starts low, increases, then plateaus
initial_reward = -15
final_reward = 80
noise = np.random.randn(len(episodes)) * 5
mean_rewards = np.linspace(initial_reward, final_reward, len(episodes)) + noise
mean_rewards = np.clip(mean_rewards, initial_reward, final_reward) # Keep rewards in a bound
# Smooth the curve a bit for a more realistic look
mean_rewards = np.convolve(mean_rewards, np.ones(3)/3, mode='valid')
episodes = episodes[:len(mean_rewards)] # Adjust episodes to new length
# --- End of Placeholder Data ---

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(episodes, mean_rewards, color='b', linewidth=2, label='Mean Reward')

# Set plot properties
ax.set_title('Figure A: Agent Training Progress', fontsize=14, weight='bold')
ax.set_xlabel('Episode Number')
ax.set_ylabel('Mean Reward per Episode')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Add a horizontal line for the target/optimal reward if you have one
ax.axhline(y=final_reward-5, color='r', linestyle='--', label='Optimal Performance Plateau')
ax.legend()

# Save the figure
plt.tight_layout()
plt.savefig('figure_agent_training_progress.png', dpi=300)
plt.show()

##############import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import numpy as np

def plot_solution(ax, title, boundary_coords, shafts, gear_train):
    """Helper function to plot a single solution."""
    # Plot boundary
    boundary_polygon = Polygon(boundary_coords)
    x, y = boundary_polygon.exterior.xy
    ax.plot(x, y, color='black', linewidth=2)
    ax.fill(x, y, color='lightgray', alpha=0.5)

    # Plot shafts
    ax.plot(shafts['input'][0], shafts['input'][1], 'ro', markersize=10, label='Input')
    ax.plot(shafts['output'][0], shafts['output'][1], 'bo', markersize=10, label='Output')

    # Plot gears
    for i, gear in enumerate(gear_train):
        center = gear['center']
        radius = gear['radius']
        gear_circle = plt.Circle(center, radius, color='steelblue', alpha=0.8, ec='black')
        ax.add_patch(gear_circle)
        ax.text(center[0], center[1], f'G{i+1}', ha='center', va='center', color='white', fontsize=8)

    # Set properties
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()


# --- Placeholder Data (Replace with your actual data) ---
BOUNDARY_COORDS = [
    (0, 0), (10, 2), (12, 8), (8, 11), (3, 10), (1, 5), (0, 0)
]
SHAFTS = {'input': (2, 7), 'output': (10, 4)}

# Solution 1: 2-Gear Sequence
GEAR_TRAIN_1 = [
    {'center': (4, 8), 'radius': 1.5},
    {'center': (7, 6), 'radius': 3.0}
]
# Solution 2: 4-Gear Sequence
GEAR_TRAIN_2 = [
    {'center': (3, 5), 'radius': 1.0},
    {'center': (5, 4), 'radius': 1.0},
    {'center': (7, 3), 'radius': 1.0},
    {'center': (9, 2.5), 'radius': 1.5}
]
# --- End of Placeholder Data ---


# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Figure B: Comparative Design Solutions', fontsize=16, weight='bold')

# Plot the first solution
plot_solution(ax1, 'Solution for 2-Gear Sequence', BOUNDARY_COORDS, SHAFTS, GEAR_TRAIN_1)

# Plot the second solution
plot_solution(ax2, 'Solution for 4-Gear Sequence', BOUNDARY_COORDS, SHAFTS, GEAR_TRAIN_2)

# Save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
plt.savefig('figure_comparative_solutions.png', dpi=300)
plt.show()
###########################
