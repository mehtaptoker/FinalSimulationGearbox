import json
import sys
import numpy as np
import os
#sys.path.append("../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.processor import Processor
from gear_generator.factory import GearFactory
from geometry_env.simulator import GearTrainSimulator
from common.data_models import Gear, Point 
from visualization.renderer import Renderer
from pathfinding.finder import Pathfinder

# Bovenaan test_gear_generation.py
from shapely.geometry import Point
from shapely.ops import unary_union
from descartes import PolygonPatch # Nodig voor de visualisatie

def create_clearance_area(path_possibilities):
    """
    Voegt alle individuele clearance-cirkels samen tot één groot oppervlak.
    """
    # Maak een lijst van alle cirkel-polygonen
    list_of_circles = [
        Point(p['point_x'], p['point_y']).buffer(p['max_radius'])
        for p in path_possibilities
    ]
    
    # Voeg alle polygonen samen
    clearance_area = unary_union(list_of_circles)
    return clearance_area

#Try to automatize the the action_space
def generate_action_space(min_teeth=10, max_teeth=50, step=5, compound=True, simple=True):
    """Genereert een lijst van mogelijke acties (driven_teeth, driving_teeth)."""
    actions = []
    # Begin met de grootste tandwielen, die hebben vaak de voorkeur
    for driven in range(max_teeth, min_teeth - 1, -step):
        for driving in range(max_teeth, min_teeth - 1, -step):
            # Voeg compound gears toe (aandrijvend en aangedreven deel verschillen)
            if compound and driven != driving:
                actions.append((driven, driving))
            # Voeg simple gears toe (aandrijvend en aangedreven deel zijn gelijk)
            elif simple and driven == driving:
                actions.append((driven, driving))
    return actions

def calculate_next_action(simulator: GearTrainSimulator, strategy_params: dict) -> tuple[int, int] | None:
    """
    Berekent een concrete actie (driven_teeth, driving_teeth) gebaseerd op
    strategische parameters, mogelijk gekozen door een RL-agent.

    Args:
        simulator: De huidige staat van de gear train simulator.
        strategy_params: Een dictionary met strategische keuzes.
            Bijv: {'size_factor': 0.9, 'compound_ratio': 0.8}

    Returns:
        Een (driven, driving) tuple of None als er geen actie mogelijk is.
    """
    last_gear = simulator.last_gear
    module = simulator.gear_factory.module

    # Controleer of we aan het einde van het pad zijn
    if simulator.current_path_index >= len(simulator.path):
        return None 

    # Bepaal het centrum van het volgende tandwiel
    next_center_point = simulator.path[simulator.current_path_index]
    
    # Bereken de afstand tot het volgende punt op het pad
    distance = np.linalg.norm(last_gear.center.to_np() - next_center_point.to_np())

    # Bereken de maximaal beschikbare straal voor het nieuwe tandwiel
    max_radius = distance - last_gear.driving_radius - simulator.clearance_margin

    # Minimale diameter moet minstens 4 tanden kunnen huisvesten
    if max_radius * 2 < 4 * module:
        return None

    # --- De "RL-Strategie" wordt hier toegepast ---
    size_factor = strategy_params.get('size_factor', 0.9) # Hoeveel van de max. ruimte gebruiken we? (0.5 - 1.0)
    compound_ratio = strategy_params.get('compound_ratio', 0.8) # Hoeveel kleiner is het aandrijvende deel? (0.5 - 1.0)
    # ----------------------------------------------

    # Kies de straal en converteer naar tanden voor het aangedreven deel
    chosen_radius = max_radius * size_factor
    driven_teeth = int((chosen_radius * 2) / module)

    # Zorg voor een minimum aantal tanden (bv. 8)
    if driven_teeth < 8:
        return None

    # Bepaal het aantal tanden voor het aandrijvende deel
    # Als ratio 1.0 is, wordt het een "simple gear"
    driving_teeth = int(driven_teeth * compound_ratio)
    if driving_teeth < 8:
        driving_teeth = driven_teeth # Behoud het aantal, zelfs als het laag is

    return (driven_teeth, driving_teeth)

def test_run_full_pipeline(fn='Example1'):
    """
    Runs the full pipeline: preprocessing and then gear generation simulation.
    """
    # 1. --- Configuration ---
    # Define directories and the base name of the example to test.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG = {
        "INPUT_DIR": os.path.join(BASE_DIR, "data"),
        "INTERMEDIATE_DIR": os.path.join(BASE_DIR, "data", "intermediate"),
        "EXAMPLE_NAME": f"{fn}",
        "module": 1.0,
        "clearance_margin": 1.0,
        "initial_gear_teeth": 20,
        "OUTPUT_DIR": os.path.join(BASE_DIR, "output"),
    }

    # 2. --- Preprocessing Step ---
    print(f"--- Running Preprocessing for: {CONFIG['EXAMPLE_NAME']} ---")
    os.makedirs(CONFIG["INTERMEDIATE_DIR"], exist_ok=True)
    input_img_path = os.path.join(CONFIG["INPUT_DIR"], f"{CONFIG['EXAMPLE_NAME']}.png")
    input_constraints_path = os.path.join(CONFIG["INPUT_DIR"], f"{CONFIG['EXAMPLE_NAME']}_constraints.json")
    processed_json_path = os.path.join(CONFIG["INTERMEDIATE_DIR"], f"{CONFIG['EXAMPLE_NAME']}_processed.json")
    
    Processor.process_input(input_img_path, input_constraints_path, processed_json_path)
    print(f"Successfully generated processed file: {processed_json_path}")

    # 3. --- Pathfinding Step ---
    print("\n--- Finding Optimal Path ---")
    path_json_path = os.path.join(CONFIG['OUTPUT_DIR'], fn, 'path.json')
    path_image_path = os.path.join(CONFIG['OUTPUT_DIR'], fn, 'path.png')
    
    # finder = Pathfinder()
    with open(path_json_path, 'r') as f:
        optimal_path = json.load(f)
    # print(optimal_path)
    # optimal_path = finder.find_path(processed_json_path)
    
    if optimal_path:
        # Save path to JSON
        # with open(path_json_path, 'w') as f:
        #     json.dump(optimal_path, f, indent=4)
        # print(f"Path saved to {path_json_path}")

        # Generate visualization
        Renderer.render_path(
            processed_json_path,
            path_image_path,
            path=optimal_path
        )
        print(f"Visualization saved to {path_image_path}")
    else:
        print("FATAL: Could not find a path between shafts.")
    
    # 4. --- Initialization ---
    print("\n--- Initializing Gear Generation Test ---")
    gear_factory = GearFactory(module=CONFIG["module"])
    with open(processed_json_path, 'r') as f:
        data = json.load(f)['normalized_space']
        path_data = data['boundaries']
        shaft_input = tuple(data['input_shaft'].values())
        shaft_output = tuple(data['output_shaft'].values())

    simulator = GearTrainSimulator(
        path=optimal_path, # <-- Use the generated optimal path
        input_shaft=shaft_input,
        output_shaft=shaft_output,
        boundaries=data['boundaries'], # Boundaries are still used for collision checks
        gear_factory=gear_factory,
        clearance_margin=CONFIG["clearance_margin"]
    )
    #toegevoegd
    next_target = simulator.path[1]  # target voor eerste intermediate gear

    # 5. --- Execution: Place Input and Intermediate Gears ---
    simulator.reset(initial_gear_teeth=CONFIG["initial_gear_teeth"])
    print(f"Step 0: Initial gear placed (ID: {simulator.last_gear.id}).")

    # Define actions for the two intermediate gears.
    # Action 1: A compound gear (40 driven teeth, 15 driving teeth).
    # Action 2: A simple gear (30 driven teeth, 30 driving teeth).
    # intermediate_actions = [
    #     (10, 15),  # Action 1: Place a compound gear
    #     (20, 15),  # Action 2: Place another compound gear
    #     (10, 10)   # Action 3: Place a simple gear
    # ]
    intermediate_actions = generate_action_space(min_teeth=10, max_teeth=40, step=5)

    done = False
    # 5. --- Execution: Place Intermediate Gears with a Dynamic Strategy ---
    simulator.reset(initial_gear_teeth=CONFIG["initial_gear_teeth"])
    print(f"Step 0: Initial gear placed (ID: {simulator.last_gear.id}).")

    done = False
    step_counter = 0
    max_steps = 10  # Voorkom een oneindige loop

    while not done and step_counter < max_steps:
        step_counter += 1
        print("-" * 20)
        print(f"Step {step_counter}: Deciding next action...")

        # ----------------------------------------------------------------------
        # HIER ZOU DE RL-AGENT EEN BESLISSING NEMEN
        # De agent observeert de 'state' en kiest de beste strategie.
        # Voor nu simuleren we dit met een vaste strategie.
        # Voorbeeld: een agent kan leren om size_factor aan te passen.
        
        # Voorbeeld van een strategie: "Wees voorzichtig, gebruik 90% van de ruimte"
        strategy = {
        "size_factor": 0.2,      # <-- Verlaagd van 0.9 naar 0.1
        "compound_ratio": 0.85
    }
        print(f"Chosen strategy: {strategy}")
        # ----------------------------------------------------------------------

        # Vertaal de strategie naar een concrete actie
        action = calculate_next_action(simulator, strategy)

        if action is None:
            print(f"FATAL: Could not calculate a valid action with strategy {strategy}.")
            break

        # Voer de berekende actie uit
        print(f"Placing intermediate gear with calculated teeth {action}...")
        state, reward, done, info = simulator.step(action)
        
        if done:
            if info.get('error'):
                print(f"Simulation failed: {info.get('error')}")
            else:
                print("Simulation finished successfully!")
        else:
            print(f"Step {step_counter}: Intermediate gear placed (ID: {simulator.last_gear.id}). Reward: {reward}")

    # ... De rest van je code (sectie 6, 7, 8) blijft grotendeels hetzelfde.
    # Let op: de logica voor het laatste tandwiel in sectie 6 is misschien niet meer nodig.

    # 6. --- Execution: Calculate and Place Final Gear (only if intermediate steps succeeded) ---
    if not done:
        print("-" * 20)
        print("Step 3: Calculating and placing final gear on output shaft...")
        last_intermediate_gear = simulator.last_gear
        
        dist_to_output = np.linalg.norm(np.array(last_intermediate_gear.center.to_np()) - np.array(shaft_output))
        final_gear_radius = dist_to_output - last_intermediate_gear.driving_radius
        
        if final_gear_radius * 2 < (8 * CONFIG["module"]): # Check if the gear would be too small
             print(f"Final gear placement failed: Required radius ({final_gear_radius:.2f}) is too small.")
        else:
            final_gear_diameter = final_gear_radius * 2
            final_gear = gear_factory.create_gear_from_diameter(
                gear_id='gear_final',
                center=shaft_output,
                desired_diameter=final_gear_diameter
            )
            simulator.gears.append(final_gear)
            print("Step 3: Final gear placed.")

    # 7. --- Save Generated Gears to JSON ---
    print("\n--- Saving generated gear train to JSON ---")
    output_dir = os.path.join(CONFIG["OUTPUT_DIR"], CONFIG["EXAMPLE_NAME"])
    os.makedirs(output_dir, exist_ok=True)
    gear_layout_path = os.path.join(output_dir, "gear_layout.json")

    # Convert gear objects to JSON-serializable dictionaries
    gears_json_data = [gear.to_json() for gear in simulator.gears]
    
    with open(gear_layout_path, 'w') as f:
        json.dump(gears_json_data, f, indent=4)
    print(f"Gear layout saved to: {gear_layout_path}")

    # 8. --- Final Visualization ---
    print("\n--- Generating final visualization from saved files ---")
    output_image_path = os.path.join(output_dir, "gear_train1_result.png")
    
    # The renderer now reads the gear layout from the JSON file
    with open(gear_layout_path, 'r') as f:
        gears_data = json.load(f)

    gears_for_renderer = [Gear.from_json(g) for g in gears_data]

    Renderer.render_processed_data(
        processed_data_path=processed_json_path,
        output_path=output_image_path,
        path=simulator.path,
        gears=gears_for_renderer
    )
    print(f"Visualization saved to: {output_image_path}")

if __name__ == "__main__":
    test_run_full_pipeline("Example2")
#JUSTIFICATION##
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from IPython.display import display, clear_output

# --- CONFIGURATIE VAN DE BEKENDE OPLOSSING ---
# We definiëren een correcte oplossing die de agent moet leren.
# Doel: een totale ratio van ~1.88 met 2 tussentandwielen.
# Input Gear: r=10
# Gear 1 (Correcte Actie 0): driven=15, driving=10 -> Ratio = 15/10 = 1.5
# Gear 2 (Correcte Actie 1): driven=12.5, driving=10 -> Ratio = 12.5/10 = 1.25
# Finale Ratio = 1.5 * 1.25 = 1.875
TARGET_GEAR_RATIO = 1.875
DESIRED_INTERMEDIATE_GEARS = 2

class Gear:
    """Dataklasse voor een tandwiel."""
    def __init__(self, id, center, driven_r, driving_r):
        self.id = id
        self.center = np.array(center)
        self.driven_radius = driven_r
        self.driving_radius = driving_r

class KnownSolutionEnv(gym.Env):
    """Een RL-omgeving die is ontworpen om te testen of de agent een specifieke,
       bekende oplossing kan leren."""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(KnownSolutionEnv, self).__init__()
        
        # Omgeving is een simpele rechthoek
        rect = [0, 0, 120, 50]
        self.boundaries = [[rect[0], rect[1]], [rect[0]+rect[2], rect[1]], [rect[0]+rect[2], rect[1]+rect[3]], [rect[0], rect[1]+rect[3]]]
        self.boundary_polygon = Polygon(self.boundaries)
        self.input_shaft_pos = np.array([15.0, 25.0])
        self.output_shaft_pos = np.array([105.0, 25.0])
        
        # --- ACTIES: De correcte keuzes + een paar "foute" keuzes ---
        self.possible_actions = [
            (15.0, 10.0),  # Correcte Actie 0
            (12.5, 10.0),  # Correcte Actie 1
            (10.0, 10.0),  # Foute Actie 2
            (8.0, 8.0),    # Foute Actie 3
        ]
        self.action_space = spaces.Discrete(len(self.possible_actions))

        # Observaties
        self.observation_space = spaces.Box(
            low=np.array([-120, -50, 0, 0]), 
            high=np.array([120, 50, 10, 10]), 
            dtype=np.float32
        )
        
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        # Houd de correcte sequentie bij om de agent te valideren
        self.correct_sequence = [0, 1] 
        self.action_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gears = []
        self.total_ratio = 1.0
        self.action_history = []
        
        input_gear = Gear("input", self.input_shaft_pos, 10.0, 10.0)
        self.gears.append(input_gear)
        self.last_gear = input_gear
        
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        vector_to_target = self.output_shaft_pos - self.last_gear.center
        return np.array([vector_to_target[0], vector_to_target[1], self.total_ratio, len(self.gears)], dtype=np.float32)

    def _get_info(self):
        return {"gears_placed": len(self.gears), "current_ratio": self.total_ratio}

    def step(self, action_index):
        self.action_history.append(action_index)
        
        # Als de agent te veel stappen zet, is het een mislukking
        if len(self.gears) > DESIRED_INTERMEDIATE_GEARS + 1:
            return self._get_obs(), -500, True, False, self._get_info()

        driven_radius, driving_radius = self.possible_actions[action_index]
        
        direction = self.output_shaft_pos - self.last_gear.center
        direction /= np.linalg.norm(direction)
        
        meshing_distance = self.last_gear.driving_radius + driven_radius
        next_center = self.last_gear.center + direction * meshing_distance
        
        new_gear_circle = Point(next_center).buffer(driven_radius)
        if not self.boundary_polygon.contains(new_gear_circle):
            return self._get_obs(), -200, True, False, self._get_info()

        new_gear = Gear(f"gear_{len(self.gears)}", next_center, driven_radius, driving_radius)
        self.gears.append(new_gear)
        self.total_ratio *= new_gear.driven_radius / self.last_gear.driving_radius
        self.last_gear = new_gear
        
        # --- FINALE REWARD BEREKENING ---
        dist_to_output = np.linalg.norm(self.last_gear.center - self.output_shaft_pos)
        final_gear_radius = dist_to_output - self.last_gear.driving_radius
        final_gear_circle = Point(self.output_shaft_pos).buffer(final_gear_radius)
        
        # Controleer of we de output kunnen bereiken
        if len(self.gears) - 1 == DESIRED_INTERMEDIATE_GEARS and self.boundary_polygon.contains(final_gear_circle):
            # We hebben het juiste aantal tandwielen. Nu controleren we de sequentie.
            if self.action_history == self.correct_sequence:
                # De agent heeft de perfecte oplossing gevonden!
                reward = 5000 
            else:
                # De agent heeft een oplossing gevonden, maar niet de juiste.
                reward = -1000
            
            final_gear = Gear("output", self.output_shaft_pos, final_gear_radius, final_gear_radius)
            self.gears.append(final_gear)
            return self._get_obs(), reward, True, False, self._get_info()
        
        return self._get_obs(), -1, False, False, self._get_info()

    def render(self):
        self.ax.clear()
        self.ax.plot(*zip(*self.boundaries, self.boundaries[0]), 'k-', linewidth=1)
        for gear in self.gears:
            self.ax.add_artist(plt.Circle(gear.center, gear.driven_radius, fc='skyblue', ec='blue', alpha=0.6))
            if gear.driven_radius != gear.driving_radius:
                self.ax.add_artist(plt.Circle(gear.center, gear.driving_radius, fc='royalblue', ec='blue'))
        self.ax.add_artist(plt.Circle(self.output_shaft_pos, 2, color='red'))
        self.ax.set_aspect('equal'); self.ax.grid(True)
        clear_output(wait=True); display(self.fig)

    def close(self):
        plt.close(self.fig)

if __name__ == "__main__":
    env = KnownSolutionEnv()
    check_env(env)

    model = PPO('MlpPolicy', env, verbose=1)
    
    TIMESTEPS = 50000
    print(f"\n--- Starting training for {TIMESTEPS} timesteps ---")
    model.learn(total_timesteps=TIMESTEPS)
    print("--- Training complete ---")

    print("\n--- Evaluating trained agent to see if it learned the correct sequence ---")
    total_successes = 0
    for episode in range(10): # Evalueer 10 keer
        obs, info = env.reset()
        done = False
        print(f"\n--- Episode {episode + 1} ---")
        for i in range(DESIRED_INTERMEDIATE_GEARS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            if done: break
        
        # Controleer of de agent de juiste acties heeft gekozen
        if env.action_history == env.correct_sequence:
            print(f"SUCCESS! Agent chose the correct sequence: {env.action_history}")
            total_successes += 1
        else:
            print(f"FAILURE. Agent chose sequence: {env.action_history}, expected: {env.correct_sequence}")
        
        env.render() # Toon het eindresultaat
    
    print(f"\nValidation complete. Agent found the correct solution in {total_successes}/10 episodes.")
    env.close()
