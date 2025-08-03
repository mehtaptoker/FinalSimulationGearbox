import json
import os
import math
from heapq import heappush, heappop
import sys
#sys.path.append("../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Installeer de benodigde bibliotheken indien nodig:
# pip install shapely
# pip install matplotlib
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# --- CONFIGURATIE ---
# Pas deze variabelen aan indien nodig
# GAAT EEN EXTRA MAP OMHOOG NAAR DE PROJECT ROOT
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_NAME = "Example2"
INPUT_DIR = os.path.join(BASE_DIR, "data")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data", "intermediate")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# --- PATHFINDER KLASSE (intern in dit script) ---
class Pathfinder:
    def find_path(self, boundaries, start_node, end_node, margin=5.0, step_size=0.5, penalty_factor=50.0):
        boundary_polygon = Polygon(boundaries).buffer(-margin)

        if not boundary_polygon.contains(Point(start_node)) or not boundary_polygon.contains(Point(end_node)):
            print("Start of eindpunt ligt buiten de toegestane zone.")
            return None

        open_set = [(0, start_node)]
        came_from = {}
        start_node_key = self._get_key(start_node)
        g_score = {start_node_key: 0}

        while open_set:
            _, current = heappop(open_set)
            current_key = self._get_key(current)

            if self._distance(current, end_node) < step_size:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._get_neighbors(current, step_size):
                neighbor_point = Point(neighbor)
                if not boundary_polygon.contains(neighbor_point):
                    continue

                distance_to_boundary = boundary_polygon.boundary.distance(neighbor_point)
                penalty = penalty_factor / (distance_to_boundary + 1e-6)
                step_cost = self._distance(current, neighbor) + penalty
                tentative_g_score = g_score.get(current_key, float('inf')) + step_cost
                
                neighbor_key = self._get_key(neighbor)

                if tentative_g_score < g_score.get(neighbor_key, float('inf')):
                    came_from[neighbor_key] = current
                    g_score[neighbor_key] = tentative_g_score
                    f_score = tentative_g_score + self._distance(neighbor, end_node)
                    heappush(open_set, (f_score, neighbor))
        return None

    def _get_key(self, node): return (round(node[0], 4), round(node[1], 4))
    def _distance(self, n1, n2): return math.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)
    def _reconstruct_path(self, came_from, current):
        path = [current]; key = self._get_key(current)
        while key in came_from:
            current = came_from[key]; key = self._get_key(current); path.append(current)
        return path[::-1]
    def _get_neighbors(self, node, step_size):
        x, y = node
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0: continue
                yield (x + dx, y + dy)

# --- HELPER FUNCTIES (intern in dit script) ---
def analyze_path_clearance(path, boundaries, margin):
    boundary_polygon = Polygon(boundaries)
    clearance_data = []
    for point_coords in path:
        point = Point(point_coords)
        max_radius = boundary_polygon.distance(point)
        adjusted_radius = max_radius - margin
        clearance_data.append({
            'point_x': point.x, 'point_y': point.y,
            'max_radius': adjusted_radius if adjusted_radius > 0 else 0
        })
    return clearance_data

def create_clearance_area(path_possibilities):
    list_of_circles = [
        Point(p['point_x'], p['point_y']).buffer(p['max_radius'])
        for p in path_possibilities if p['max_radius'] > 0
    ]
    return unary_union(list_of_circles) if list_of_circles else None

def plot_shape(ax, geom, **kwargs):
    if geom is None or geom.is_empty: return
    if hasattr(geom, 'geoms'):
        for sub_geom in geom.geoms:
            plot_shape(ax, sub_geom, **kwargs)
        return
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        ax.fill(x, y, **kwargs)
        for interior in geom.interiors:
            x_i, y_i = interior.xy
            ax.fill(x_i, y_i, fc='white', ec='none')

# --- HOOFD LOGICA ---
if __name__ == "__main__":
    print(f"--- Starting analysis for {EXAMPLE_NAME} ---")

    # 1. Laad de data
    processed_json_path = os.path.join(INTERMEDIATE_DIR, f"{EXAMPLE_NAME}_processed.json")
    with open(processed_json_path, 'r') as f:
        data = json.load(f)['normalized_space']
    
    boundaries = data['boundaries']
    start_node = tuple(data['input_shaft'].values())
    end_node = tuple(data['output_shaft'].values())

    # 2. Vind het pad
    print("Finding optimal path...")
    pathfinder = Pathfinder()
    # U kunt de margin en penalty_factor hier aanpassen voor een ander pad
    optimal_path = pathfinder.find_path(boundaries, start_node, end_node, margin=1.0, penalty_factor=20.0)

    if not optimal_path:
        print("FATAL: Could not find a path.")
    else:
        print("Path found. Analyzing clearance...")
        
        # 3. Analyseer en creëer de clearance area
        path_possibilities = analyze_path_clearance(optimal_path, boundaries, margin=1.0)
        clearance_area = create_clearance_area(path_possibilities)

        # 4. Sla het pad op voor later gebruik
        output_dir_example = os.path.join(OUTPUT_DIR, EXAMPLE_NAME)
        os.makedirs(output_dir_example, exist_ok=True)
        path_json_path = os.path.join(output_dir_example, f'{EXAMPLE_NAME}_path.json')
        with open(path_json_path, 'w') as f:
            json.dump(optimal_path, f, indent=4)
        print(f"Path saved to {path_json_path}")

        # 5. Visualiseer het resultaat
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Teken grenzen en assen
        ax.plot(*zip(*boundaries, boundaries[0]), 'k-', label='Boundary')
        ax.plot(start_node[0], start_node[1], 'ro', markersize=10, label='Input Shaft')
        ax.plot(end_node[0], end_node[1], 'bo', markersize=10, label='Output Shaft')

        # Teken clearance area en pad
        plot_shape(ax, clearance_area, face_color='cyan', alpha=0.5)
        ax.plot(*zip(*optimal_path), 'm-', label='Generated Path')

        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        ax.legend()
        
        output_image_path = os.path.join(output_dir_example, f'{EXAMPLE_NAME}_clearance.png')
        plt.savefig(output_image_path, dpi=200)
        print(f"Clearance visualization saved to {output_image_path}")
        plt.close(fig)

    print("--- Analysis complete ---")
    # Inhoud voor: visualize_clearance_from_file.py

import json
import os
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# --- CONFIGURATIE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_NAME = "Example2"
MARGIN = 1.0

# Paden naar de input-bestanden
PROCESSED_JSON_PATH = os.path.join(BASE_DIR, "data", "intermediate", f"{EXAMPLE_NAME}_processed.json")
PATH_JSON_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, "path.json")

# Pad naar de output-afbeelding
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, "clearance_from_file.png")


# --- HELPER FUNCTIES ---
def analyze_clearance(path, boundaries, margin):
    boundary_polygon = Polygon(boundaries)
    clearance_data = []
    for point_coords in path:
        point = Point(point_coords)
        max_radius = boundary_polygon.distance(point)
        adjusted_radius = max_radius - margin
        clearance_data.append({
            'point_x': point.x, 'point_y': point.y,
            'max_radius': adjusted_radius if adjusted_radius > 0 else 0
        })
    return clearance_data

def create_area(possibilities):
    list_of_circles = [
        Point(p['point_x'], p['point_y']).buffer(p['max_radius'])
        for p in possibilities if p['max_radius'] > 0
    ]
    return unary_union(list_of_circles) if list_of_circles else None

def plot_shape(ax, geom, **kwargs):
    if geom is None or geom.is_empty: return
    if hasattr(geom, 'geoms'):
        for sub_geom in geom.geoms:
            plot_shape(ax, sub_geom, **kwargs)
        return
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        ax.fill(x, y, **kwargs)
        for interior in geom.interiors:
            x_i, y_i = interior.xy
            ax.fill(x_i, y_i, fc='white', ec='none')


# --- HOOFD LOGICA ---
if __name__ == "__main__":
    print(f"--- Starting clearance analysis for path from file: {PATH_JSON_PATH} ---")

    # 1. Laad de boundary en shaft data
    with open(PROCESSED_JSON_PATH, 'r') as f:
        data = json.load(f)['normalized_space']
    boundaries = data['boundaries']
    start_node = tuple(data['input_shaft'].values())
    end_node = tuple(data['output_shaft'].values())

    # 2. Laad het pad uit het JSON bestand
    try:
        with open(PATH_JSON_PATH, 'r') as f:
            loaded_path = json.load(f)
        print("Successfully loaded path from file.")
    except FileNotFoundError:
        print(f"FATAL: Path file not found at {PATH_JSON_PATH}")
        exit()

    # 3. Analyseer en creëer de clearance area
    path_possibilities = analyze_clearance(loaded_path, boundaries, MARGIN)
    clearance_area = create_area(path_possibilities)
    print("Clearance analysis complete.")

    # 4. Visualiseer het resultaat
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Teken grenzen en assen
    ax.plot(*zip(*boundaries, boundaries[0]), 'k-', linewidth=2, label='Boundary')
    ax.plot(start_node[0], start_node[1], 'ro', markersize=10, label='Input Shaft')
    ax.plot(end_node[0], end_node[1], 'bo', markersize=10, label='Output Shaft')

    # Teken de berekende clearance area
    plot_shape(ax, clearance_area, face_color='cyan', alpha=0.5)
    
    # Teken het geladen pad
    ax.plot(*zip(*loaded_path), 'm-', linewidth=2, label='Loaded Path')

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()
    
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=200)
    print(f"Clearance visualization for loaded path saved to: {OUTPUT_IMAGE_PATH}")
    plt.close(fig)

    # Inhoud voor: analyze_path_width.py

import json
import os
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt

# --- CONFIGURATIE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_NAME = "Example2"
STEP_DISTANCE = 1.0  # Analyseer het pad elke 1 mm

# Paden naar de input-bestanden
PROCESSED_JSON_PATH = os.path.join(BASE_DIR, "data", "intermediate", f"{EXAMPLE_NAME}_processed.json")
PATH_JSON_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, "path.json")

# Pad naar de output-afbeelding
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, "path_width_analysis.png")


# --- HELPER FUNCTIES ---

def interpolate_path(path, step_distance):
    """Maakt een pad met punten op een vaste afstand van elkaar."""
    path_points = np.array(path)
    distances = np.sqrt(np.sum(np.diff(path_points, axis=0)**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    
    total_length = cumulative_dist[-1]
    num_steps = int(total_length / step_distance)
    
    interpolated_distances = np.linspace(0, total_length, num_steps)
    interpolated_points = np.empty((num_steps, 2))
    
    interpolated_points[:, 0] = np.interp(interpolated_distances, cumulative_dist, path_points[:, 0])
    interpolated_points[:, 1] = np.interp(interpolated_distances, cumulative_dist, path_points[:, 1])
    
    return interpolated_points

# --- HOOFD LOGICA ---
if __name__ == "__main__":
    print(f"--- Starting path width analysis for: {PATH_JSON_PATH} ---")

    # 1. Laad de data
    with open(PROCESSED_JSON_PATH, 'r') as f:
        data = json.load(f)['normalized_space']
    boundaries = data['boundaries']
    boundary_polygon = Polygon(boundaries)

    try:
        with open(PATH_JSON_PATH, 'r') as f:
            loaded_path = json.load(f)
        print("Successfully loaded path from file.")
    except FileNotFoundError:
        print(f"FATAL: Path file not found at {PATH_JSON_PATH}")
        exit()

    # 2. Creëer een gedetailleerd pad
    dense_path = interpolate_path(loaded_path, STEP_DISTANCE)
    print(f"Path interpolated into {len(dense_path)} steps.")

    # 3. Analyseer de breedte op elk punt
    width_lines = []
    max_width = 0.0
    
    for i in range(1, len(dense_path) - 1):
        # Bepaal de richting van het pad
        p_prev = dense_path[i-1]
        p_next = dense_path[i+1]
        p_current = dense_path[i]
        
        direction_vector = p_next - p_prev
        # Normaliseer de vector
        norm = np.linalg.norm(direction_vector)
        if norm == 0: continue
        direction_vector /= norm
        
        # Bepaal de loodrechte (normaal) vector
        normal_vector = np.array([-direction_vector[1], direction_vector[0]])
        
        # Maak een hele lange lijn loodrecht op het pad
        line_start = p_current - normal_vector * 1000
        line_end = p_current + normal_vector * 1000
        long_line = LineString([line_start, line_end])
        
        # Vind de intersectie met de grenzen
        intersection = boundary_polygon.intersection(long_line)
        
        if not intersection.is_empty and (intersection.geom_type == 'LineString' or intersection.geom_type == 'MultiLineString'):
            width_lines.append(intersection)
            if intersection.length > max_width:
                max_width = intersection.length

    print(f"\nMaximum width found: {max_width:.2f} mm")

    # 4. Visualiseer het resultaat
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Teken grenzen en het originele pad
    ax.plot(*zip(*boundaries, boundaries[0]), 'k-', linewidth=2, label='Boundary')
    ax.plot(*zip(*loaded_path), 'm-', linewidth=2, label='Original Path')

    # Teken alle breedte-lijnen
    for line in width_lines:
        if line.geom_type == 'LineString':
            ax.plot(*line.xy, 'c-', linewidth=0.5)
        elif line.geom_type == 'MultiLineString':
            for part in line.geoms:
                ax.plot(*part.xy, 'c-', linewidth=0.5)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()
    
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=200)
    print(f"Path width analysis saved to: {OUTPUT_IMAGE_PATH}")
    plt.close(fig)
# Inhoud voor: analyze_path_width.py

import json
import os
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt

# --- CONFIGURATIE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_NAME = "Example2"
STEP_DISTANCE = 1.0  # Analyseer het pad elke 1 mm

# Paden naar de input-bestanden
PROCESSED_JSON_PATH = os.path.join(BASE_DIR, "data", "intermediate", f"{EXAMPLE_NAME}_processed.json")
PATH_JSON_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, "path.json")

# Pad naar de output-afbeelding
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, "path_width_analysis.png")


# --- HELPER FUNCTIES ---

def interpolate_path(path, step_distance):
    """Maakt een pad met punten op een vaste afstand van elkaar."""
    path_points = np.array(path)
    distances = np.sqrt(np.sum(np.diff(path_points, axis=0)**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    
    total_length = cumulative_dist[-1]
    num_steps = int(total_length / step_distance)
    
    interpolated_distances = np.linspace(0, total_length, num_steps)
    interpolated_points = np.empty((num_steps, 2))
    
    interpolated_points[:, 0] = np.interp(interpolated_distances, cumulative_dist, path_points[:, 0])
    interpolated_points[:, 1] = np.interp(interpolated_distances, cumulative_dist, path_points[:, 1])
    
    return interpolated_points

def calculate_path_length(path):
    """Berekent de totale lengte van een pad."""
    length = 0.0
    path_points = np.array(path)
    # Berekent de afstanden tussen alle opeenvolgende punten en telt ze op
    distances = np.sqrt(np.sum(np.diff(path_points, axis=0)**2, axis=1))
    length = np.sum(distances)
    return length

# --- HOOFD LOGICA ---
if __name__ == "__main__":
    print(f"--- Starting path width analysis for: {PATH_JSON_PATH} ---")

    # 1. Laad de data
    with open(PROCESSED_JSON_PATH, 'r') as f:
        data = json.load(f)['normalized_space']
    boundaries = data['boundaries']
    boundary_polygon = Polygon(boundaries)

    try:
        with open(PATH_JSON_PATH, 'r') as f:
            loaded_path = json.load(f)
        print("Successfully loaded path from file.")
    except FileNotFoundError:
        print(f"FATAL: Path file not found at {PATH_JSON_PATH}")
        exit()

    # 2. Bereken en print de padlengte
    total_path_length = calculate_path_length(loaded_path)
    print(f"\nTotal path length: {total_path_length:.2f} mm")

    # 3. Creëer een gedetailleerd pad voor de breedte-analyse
    dense_path = interpolate_path(loaded_path, STEP_DISTANCE)
    print(f"Path interpolated into {len(dense_path)} steps for width analysis.")

    # 4. Analyseer de breedte op elk punt
    width_lines = []
    max_width = 0.0
    
    for i in range(1, len(dense_path) - 1):
        # Bepaal de richting van het pad
        p_prev = dense_path[i-1]
        p_next = dense_path[i+1]
        p_current = dense_path[i]
        
        direction_vector = p_next - p_prev
        # Normaliseer de vector
        norm = np.linalg.norm(direction_vector)
        if norm == 0: continue
        direction_vector /= norm
        
        # Bepaal de loodrechte (normaal) vector
        normal_vector = np.array([-direction_vector[1], direction_vector[0]])
        
        # Maak een hele lange lijn loodrecht op het pad
        line_start = p_current - normal_vector * 1000
        line_end = p_current + normal_vector * 1000
        long_line = LineString([line_start, line_end])
        
        # Vind de intersectie met de grenzen
        intersection = boundary_polygon.intersection(long_line)
        
        if not intersection.is_empty and (intersection.geom_type == 'LineString' or intersection.geom_type == 'MultiLineString'):
            width_lines.append(intersection)
            if intersection.length > max_width:
                max_width = intersection.length

    print(f"Maximum width found: {max_width:.2f} mm")

    # 5. Visualiseer het resultaat
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Teken grenzen en het originele pad
    ax.plot(*zip(*boundaries, boundaries[0]), 'k-', linewidth=2, label='Boundary')
    ax.plot(*zip(*loaded_path), 'm-', linewidth=2, label='Original Path')

    # Teken alle breedte-lijnen
    for line in width_lines:
        if line.geom_type == 'LineString':
            ax.plot(*line.xy, 'c-', linewidth=0.5)
        elif line.geom_type == 'MultiLineString':
            for part in line.geoms:
                ax.plot(*part.xy, 'c-', linewidth=0.5)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()
    
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=200)
    print(f"Path width analysis saved to: {OUTPUT_IMAGE_PATH}")
    plt.close(fig)
    import json
import os
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt

# --- CONFIGURATIE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_NAME = "Example2"
STEP_DISTANCE = 1.0  # Analyseer het pad elke 1 mm
MARGIN = 1.0 # De marge die van de breedte wordt afgetrokken

# Input bestanden
PROCESSED_JSON_PATH = os.path.join(BASE_DIR, "data", "intermediate", f"{EXAMPLE_NAME}_processed.json")
PATH_JSON_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, "path.json")

# Output bestanden
OUTPUT_DATA_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, "path_coordinate_system.json")
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, "path_coordinate_system.png")


# --- HELPER FUNCTIES ---

def interpolate_path(path, step_distance):
    """Maakt een pad met punten op een vaste afstand van elkaar."""
    path_points = np.array(path)
    # Bereken de afstanden tussen opeenvolgende punten
    distances = np.sqrt(np.sum(np.diff(path_points, axis=0)**2, axis=1))
    # Creëer een cumulatieve afstandstabel (0.0, 5.2, 11.8, ...)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    
    total_length = cumulative_dist[-1]
    # Bepaal het aantal stappen op basis van de gewenste afstand
    num_steps = int(total_length / step_distance)
    
    # Creëer de nieuwe afstands-punten voor de interpolatie
    interpolated_distances = np.linspace(0, total_length, num_steps)
    
    # Interpoleren van x en y coördinaten
    interp_x = np.interp(interpolated_distances, cumulative_dist, path_points[:, 0])
    interp_y = np.interp(interpolated_distances, cumulative_dist, path_points[:, 1])
    
    # Combineer de geïnterpoleerde punten en afstanden
    interpolated_points = np.column_stack((interp_x, interp_y))
    
    return interpolated_points, interpolated_distances

# --- HOOFD LOGICA ---
if __name__ == "__main__":
    print(f"--- Exporting path coordinate system for: {PATH_JSON_PATH} ---")

    # 1. Laad de data
    with open(PROCESSED_JSON_PATH, 'r') as f:
        data = json.load(f)['normalized_space']
    boundaries = data['boundaries']
    boundary_polygon = Polygon(boundaries)

    try:
        with open(PATH_JSON_PATH, 'r') as f:
            loaded_path = json.load(f)
        print("Successfully loaded path from file.")
    except FileNotFoundError:
        print(f"FATAL: Path file not found at {PATH_JSON_PATH}")
        exit()

    # 2. Creëer het gedetailleerde pad (coördinatensysteem)
    dense_path, cumulative_distances = interpolate_path(loaded_path, STEP_DISTANCE)
    print(f"Path interpolated into {len(dense_path)} steps.")

    # 3. Analyseer de breedte op elk punt en bouw de data op
    coordinate_system_data = []
    
    for i in range(len(dense_path)):
        p_current = dense_path[i]
        
        # Bepaal de richting van het pad op dit punt
        if i == 0:
            direction_vector = dense_path[i+1] - p_current
        elif i == len(dense_path) - 1:
            direction_vector = p_current - dense_path[i-1]
        else:
            direction_vector = dense_path[i+1] - dense_path[i-1]
        
        norm = np.linalg.norm(direction_vector)
        if norm == 0: continue
        direction_vector /= norm
        
        # Bepaal de loodrechte (normaal) vector
        normal_vector = np.array([-direction_vector[1], direction_vector[0]])
        
        # Maak een lange lijn en vind de intersectie
        long_line = LineString([p_current - normal_vector * 1000, p_current + normal_vector * 1000])
        intersection = boundary_polygon.intersection(long_line)
        
        width = 0.0
        if not intersection.is_empty and (intersection.geom_type == 'LineString'):
            width = intersection.length

        # Voeg de data toe aan ons coördinatensysteem
        coordinate_system_data.append({
            "distance_along_path": cumulative_distances[i],
            "x": p_current[0],
            "y": p_current[1],
            "max_width": width - (2 * MARGIN) # Trek aan beide kanten de marge eraf
        })

    # 4. Sla het coördinatensysteem op
    with open(OUTPUT_DATA_PATH, 'w') as f:
        json.dump(coordinate_system_data, f, indent=4)
    print(f"\nPath coordinate system data saved to: {OUTPUT_DATA_PATH}")

    # 5. Visualiseer (optioneel, om te controleren)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(*zip(*boundaries, boundaries[0]), 'k-', linewidth=1, label='Boundary')
    
    # Teken de punten van het nieuwe coördinatensysteem
    coords_x = [p['x'] for p in coordinate_system_data]
    coords_y = [p['y'] for p in coordinate_system_data]
    widths = [p['max_width'] for p in coordinate_system_data]
    
    sc = ax.scatter(coords_x, coords_y, c=widths, cmap='viridis', s=10, label='Path Width (mm)')
    plt.colorbar(sc, ax=ax)
    
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()
    
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=200)
    print(f"Visualization of coordinate system saved to: {OUTPUT_IMAGE_PATH}")
    plt.close(fig)



####last try

# Inhoud voor: place_gears_strategically.py

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATIE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_NAME = "Example2"
MODULE = 1.0
MIN_TEETH = 8
# 'min_gears' (groot) of 'max_gears' (klein)
STRATEGY = 'max_gears' 

# Input/Output Paden
COORDS_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, "path_coordinate_system.json")
PROCESSED_JSON_PATH = os.path.join(BASE_DIR, "data", "intermediate", f"{EXAMPLE_NAME}_processed.json")
OUTPUT_LAYOUT_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, f"strategic_gear_layout_{STRATEGY}.json")
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, f"strategic_gear_train_{STRATEGY}.png")
# --- DATA MODELLEN ---
# --- DATA MODELLEN ---
class Gear:
    def __init__(self, gear_id, center, driven_radius, driving_radius=None):
        self.id = gear_id
        self.center = np.array(center)
        self.driven_radius = driven_radius
        self.driving_radius = driving_radius if driving_radius is not None else driven_radius
    
    def to_json(self):
        return {"id": self.id, "center": {"x": self.center[0], "y": self.center[1]}, "driven_radius": self.driven_radius, "driving_radius": self.driving_radius}

# --- HOOFD LOGICA ---
if __name__ == "__main__":
    print(f"--- Starting strategic gear placement with strategy: '{STRATEGY}' ---")

    # 1. Laad data
    try:
        with open(COORDS_PATH, 'r') as f: coords_data = json.load(f)
        with open(PROCESSED_JSON_PATH, 'r') as f: processed_data = json.load(f)['normalized_space']
    except FileNotFoundError as e:
        print(f"FATAL: Could not find data file: {e.filename}"); exit()

    boundaries = processed_data['boundaries']
    boundary_polygon = Polygon(boundaries)
    input_shaft_pos = np.array(list(processed_data['input_shaft'].values()))
    output_shaft_pos = np.array(list(processed_data['output_shaft'].values()))
    
    # 2. Initialiseer
    placed_gears = []
    # Creëer een shapely cirkel voor het input tandwiel en controleer of het past
    input_gear_radius = coords_data[0]['max_width'] / 2
    input_gear_circle = Point(input_shaft_pos).buffer(input_gear_radius)
    if not boundary_polygon.contains(input_gear_circle):
        # Als het niet past, verklein het dan tot het wel past
        input_gear_radius = boundary_polygon.distance(Point(input_shaft_pos))
        
    input_gear = Gear("gear_input", input_shaft_pos, input_gear_radius)
    placed_gears.append(input_gear)
    
    last_gear = input_gear
    gear_counter = 1
    print(f"Placed input gear with radius {input_gear_radius:.2f}")

    # 3. Loop om TUSSENTANDWIELEN te plaatsen
    while True:
        print(f"\n--- Step {gear_counter} ---")
        
        dist_from_last_to_output = np.linalg.norm(last_gear.center - output_shaft_pos)
        if dist_from_last_to_output <= last_gear.driving_radius + (MIN_TEETH * MODULE / 2):
             print("Too close to output shaft. Moving to final gear placement.")
             break

        action_found = False
        size_factors = np.arange(0.2, 1.0, 0.05) if STRATEGY == 'max_gears' else np.arange(0.95, 0.1, -0.05)

        # Vind het punt op het pad dat het dichtst bij het centrum van het laatste tandwiel ligt
        path_points = np.array([[p['x'], p['y']] for p in coords_data])
        distances_to_path = np.linalg.norm(path_points - last_gear.center, axis=1)
        current_path_idx = np.argmin(distances_to_path)

        # Bepaal de lokale richting van het pad
        if current_path_idx + 1 >= len(path_points): break
        direction_vector = path_points[current_path_idx + 1] - path_points[current_path_idx]
        norm = np.linalg.norm(direction_vector)
        if norm == 0: continue
        direction_vector /= norm # Genormaliseerde richtingsvector

        for size_factor in size_factors:
            # Bepaal de grootte van het tandwiel dat we proberen te plaatsen
            max_allowed_radius_here = coords_data[current_path_idx]['max_width'] / 2
            driven_radius = max_allowed_radius_here * size_factor
            
            if (driven_radius * 2 / MODULE) < MIN_TEETH: continue
            
            driving_radius = driven_radius * 0.8
            
            # Bereken de nieuwe positie PUUR GEOMETRISCH
            meshing_distance = last_gear.driving_radius + driven_radius
            next_center = last_gear.center + direction_vector * meshing_distance
            
            # *** NIEUWE ROBUUSTE VALIDATIE ***
            # Creëer een shapely cirkel voor het nieuwe tandwiel
            new_gear_circle = Point(next_center).buffer(driven_radius)
            
            # Valideer of de cirkel volledig binnen de grenzen past
            if boundary_polygon.contains(new_gear_circle):
                print(f"  Found valid intermediate step. Radius: {driven_radius:.2f}")
                new_gear = Gear(f"gear_{gear_counter}", next_center, driven_radius, driving_radius)
                placed_gears.append(new_gear)
                last_gear = new_gear
                gear_counter += 1
                action_found = True
                break
        
        if not action_found:
            print("Stopping: Could not find any fitting intermediate gear.")
            break

    # 4. Plaats het LAATSTE TANDWIEL
    print("\n--- Placing Final Gear ---")
    dist_to_output = np.linalg.norm(last_gear.center - output_shaft_pos)
    final_gear_radius = dist_to_output - last_gear.driving_radius
    
    # *** NIEUWE ROBUUSTE VALIDATIE ***
    final_gear_circle = Point(output_shaft_pos).buffer(final_gear_radius)
    
    if (final_gear_radius * 2 / MODULE) >= MIN_TEETH and boundary_polygon.contains(final_gear_circle):
        print(f"Calculated final gear radius: {final_gear_radius:.2f}")
        output_gear = Gear("gear_output", output_shaft_pos, final_gear_radius)
        placed_gears.append(output_gear)
        print("Successfully placed final gear.")
    else:
        print(f"Could not place final gear. Needed radius: {final_gear_radius:.2f}")

    # 5. Opslaan en Visualiseren
    with open(OUTPUT_LAYOUT_PATH, 'w') as f: json.dump([g.to_json() for g in placed_gears], f, indent=4)
    print(f"\nStrategic gear layout saved to: {OUTPUT_LAYOUT_PATH}")

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(*zip(*boundaries, boundaries[0]), 'k-', linewidth=1, label='Boundary')
    for gear in placed_gears:
        ax.add_artist(plt.Circle(gear.center, gear.driven_radius, facecolor='skyblue', edgecolor='blue', alpha=0.6))
        if gear.driven_radius != gear.driving_radius:
             ax.add_artist(plt.Circle(gear.center, gear.driving_radius, facecolor='royalblue', edgecolor='blue'))
        ax.text(gear.center[0], gear.center[1], gear.id, ha='center', va='center', fontsize=8)

    ax.set_aspect('equal', adjustable='box'); ax.grid(True); ax.legend()
    plt.title(f"Strategically Placed Gear Train (Strategy: {STRATEGY})")
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=200)
    print(f"Final visualization saved to: {OUTPUT_IMAGE_PATH}"); plt.close(fig)
# generate_thesis_visuals.py
#
# This script generates a series of publication-quality visuals
# to explain the methodology of the AI-powered gearbox design project.

# --- Step 1: Install necessary libraries (uncomment in Colab) ---
# !pip install stable-baselines3[extra] torch gymnasium matplotlib shapely numpy

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon
from IPython.display import display, clear_output

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

print("All libraries imported successfully.")

# --- Create a directory to save the visuals ---
OUTPUT_DIR = "thesis_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Visual 1: The Problem Definition ---
def generate_problem_definition_visual():
    """Generates a visual that clearly outlines the input constraints."""
    print("Generating Visual 1: Problem Definition...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define the complex shape
    boundaries = [[-10,0],[-10,20],[0,30],[50,30],[60,20],[60,0],[50,-10],[0,-10],[-10,0]]
    input_shaft = [0.0, 10.0]
    output_shaft = [50.0, 15.0]

    # Plot the geometry
    ax.plot(*zip(*boundaries, boundaries[0]), 'k-', linewidth=2, label='Geometric Boundary')
    ax.plot(input_shaft[0], input_shaft[1], 'go', markersize=12, label='Input Shaft')
    ax.plot(output_shaft[0], output_shaft[1], 'ro', markersize=12, label='Output Shaft')

    # Add annotations
    ax.annotate('Geometric Constraint:\nComplex Housing Shape', xy=(25, 31), xytext=(5, 35),
                arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
    
    ax.annotate('Fixed Start/End Points', xy=(25, 12.5), xytext=(25, 5),
                arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

    # Add text box for physical constraints
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    phys_text = "Physical Requirements:\n- Target Ratio: 2.0\n- Sequence: 3 Gears"
    ax.text(0.65, 0.25, phys_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ax.set_title('Visual 1: The Problem Definition', fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "1_problem_definition.png"), dpi=300)
    plt.close(fig)
    print("Visual 1 saved.")

# --- Visual 2: The Analysis Pipeline ---
def generate_analysis_pipeline_visual():
    """Generates a visual representing the data analysis pipeline."""
    print("Generating Visual 2: Analysis Pipeline...")
    # This visual is a flowchart, best created in a diagram tool.
    # Here, we generate a key output of the pipeline: the clearance area.
    
    # Re-using parts of the RL environment's setup for consistency
    from shapely.ops import unary_union
    
    boundaries = [[-10,0],[-10,20],[0,30],[50,30],[60,20],[60,0],[50,-10],[0,-10],[-10,0]]
    boundary_poly = Polygon(boundaries)
    
    # Simplified path for visualization purposes
    path_points = np.array([
        [0, 10], [10, 15], [25, 18], [40, 17], [50, 15]
    ])
    
    # Calculate clearance
    clearance_circles = []
    for point in path_points:
        radius = boundary_poly.distance(Point(point)) - 1.0 # 1.0 margin
        if radius > 0:
            clearance_circles.append(Point(point).buffer(radius))
    
    clearance_area = unary_union(clearance_circles)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(*zip(*boundaries, boundaries[0]), 'k-', linewidth=2, label='Boundary')
    ax.plot(path_points[:,0], path_points[:,1], 'm--', linewidth=2, label='Optimal Path')
    
    if hasattr(clearance_area, 'geoms'):
        for geom in clearance_area.geoms:
            ax.add_patch(patches.Polygon(np.array(geom.exterior.coords), facecolor='cyan', alpha=0.5))
    else:
        ax.add_patch(patches.Polygon(np.array(clearance_area.exterior.coords), facecolor='cyan', alpha=0.5))

    ax.set_title('Visual 2: Output of Analysis - Clearance Area', fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "2_analysis_pipeline_output.png"), dpi=300)
    plt.close(fig)
    print("Visual 2 saved.")

# --- Visual 3: The Training Progress Graph ---
def generate_training_graph_visual():
    """Generates a simulated training progress graph."""
    print("Generating Visual 3: Training Progress Graph...")
    
    # Simulate training data: mean reward should generally increase over time
    timesteps = np.linspace(0, 50000, 100)
    # Start with a low/negative reward and improve, adding some noise
    base_reward = -200 + 1500 * (1 - np.exp(-timesteps/20000))
    noise = np.random.normal(0, 50, 100)
    ep_rew_mean = base_reward + noise
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, ep_rew_mean, label='Mean Reward per Episode')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Visual 3: Agent Learning Progress (Simulated)', fontsize=16)
    ax.grid(True)
    ax.legend()
    
    # Add annotation to explain the trend
    ax.annotate('Agent discovers the\nreward for correct ratio', xy=(30000, 1100), xytext=(5000, 500),
                arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "3_training_progress.png"), dpi=300)
    plt.close(fig)
    print("Visual 3 saved.")

# --- Visual 4 & 5: Comparison of Different Solutions ---
# We need the full RL environment to generate these solutions.
class Gear:
    def __init__(self, id, center, driven_r, driving_r):
        self.id, self.center, self.driven_radius, self.driving_radius = id, np.array(center), driven_r, driving_r

class GearboxEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, desired_gears=2):
        super(GearboxEnv, self).__init__()
        self.DESIRED_INTERMEDIATE_GEARS = desired_gears
        rect = [0, 0, 100, 40]
        self.boundaries = [[rect[0], rect[1]], [rect[0]+rect[2], rect[1]], [rect[0]+rect[2], rect[1]+rect[3]], [rect[0], rect[1]+rect[3]]]
        self.boundary_polygon = Polygon(self.boundaries)
        self.input_shaft_pos = np.array([10.0, 20.0])
        self.output_shaft_pos = np.array([90.0, 20.0])
        self.possible_actions = [5.0, 8.0, 10.0, 12.0, 15.0]
        self.action_space = spaces.Discrete(len(self.possible_actions))
        self.observation_space = spaces.Box(low=np.array([-100,-40,0,0]), high=np.array([100,40,10,10]), dtype=np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gears = []
        self.total_ratio = 1.0
        input_gear = Gear("input", self.input_shaft_pos, 10.0, 10.0)
        self.gears.append(input_gear)
        self.last_gear = input_gear
        return self._get_obs(), {}
    def _get_obs(self):
        vec = self.output_shaft_pos - self.last_gear.center
        return np.array([vec[0], vec[1], self.total_ratio, len(self.gears)], dtype=np.float32)
    def step(self, action_index):
        if len(self.gears) > self.DESIRED_INTERMEDIATE_GEARS + 1: return self._get_obs(), -500, True, False, {}
        driven_radius = self.possible_actions[action_index]
        direction = (self.output_shaft_pos - self.last_gear.center)
        direction /= np.linalg.norm(direction)
        driving_radius = driven_radius * 0.8
        meshing_dist = self.last_gear.driving_radius + driven_radius
        next_center = self.last_gear.center + direction * meshing_dist
        new_gear_circle = Point(next_center).buffer(driven_radius)
        if not self.boundary_polygon.contains(new_gear_circle): return self._get_obs(), -200, True, False, {}
        new_gear = Gear(f"g_{len(self.gears)}", next_center, driven_radius, driving_radius)
        self.gears.append(new_gear)
        self.total_ratio *= new_gear.driven_radius / self.last_gear.driving_radius
        self.last_gear = new_gear
        dist_to_output = np.linalg.norm(self.last_gear.center - self.output_shaft_pos)
        final_r = dist_to_output - self.last_gear.driving_radius
        final_c = Point(self.output_shaft_pos).buffer(final_r)
        if (final_r * 2 / 1.0) >= 8 and self.boundary_polygon.contains(final_c):
            final_ratio = self.total_ratio * (final_r / self.last_gear.driving_radius)
            err = abs(final_ratio - 2.0) / 2.0
            reward = 1000 * (1 - err)
            if len(self.gears) - 1 == self.DESIRED_INTERMEDIATE_GEARS: reward += 500
            else: reward -= 500
            final_gear = Gear("output", self.output_shaft_pos, final_r, final_r)
            self.gears.append(final_gear)
            return self._get_obs(), reward, True, False, {}
        return self._get_obs(), -1, False, False, {}
    def render_to_ax(self, ax):
        ax.clear()
        ax.plot(*zip(*self.boundaries, self.boundaries[0]), 'k-', linewidth=1)
        for gear in self.gears:
            ax.add_artist(plt.Circle(gear.center, gear.driven_radius, fc='skyblue', ec='blue', alpha=0.6))
            if gear.driven_radius != gear.driving_radius:
                ax.add_artist(plt.Circle(gear.center, gear.driving_radius, fc='royalblue', ec='blue'))
        ax.add_artist(plt.Circle(self.output_shaft_pos, 2, color='red'))
        ax.set_aspect('equal'); ax.grid(True)

def generate_solution_visual(desired_gears, filename):
    """Trains an agent for a specific goal and saves the visual."""
    print(f"Generating solution for {desired_gears} intermediate gears...")
    env = GearboxEnv(desired_gears=desired_gears)
    model = PPO('MlpPolicy', env, n_steps=1024)
    model.learn(total_timesteps=30000)
    
    obs, info = env.reset()
    done = False
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        if done: break
    
    # Plot final state
    fig, ax = plt.subplots(figsize=(10, 6))
    env.render_to_ax(ax)
    ax.set_title(f'Solution with {desired_gears} Intermediate Gears', fontsize=14)
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved solution to {filename}")

def generate_comparison_visual():
    """Creates a side-by-side comparison of two different solutions."""
    print("Generating Visual 4: Comparison of Solutions...")
    
    # Generate the two separate solution images first
    solution_2_gears_path = os.path.join(OUTPUT_DIR, "temp_solution_2_gears.png")
    solution_4_gears_path = os.path.join(OUTPUT_DIR, "temp_solution_4_gears.png")
    
    generate_solution_visual(2, solution_2_gears_path)
    generate_solution_visual(4, solution_4_gears_path)
    
    # Combine them into one figure
    img2 = plt.imread(solution_2_gears_path)
    img4 = plt.imread(solution_4_gears_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    ax1.imshow(img2)
    ax1.set_title('Agent Result for Target: 2 Intermediate Gears', fontsize=16)
    ax1.axis('off')
    
    ax2.imshow(img4)
    ax2.set_title('Agent Result for Target: 4 Intermediate Gears', fontsize=16)
    ax2.axis('off')
    
    plt.suptitle('Visual 4: Agent Adapts to Different Sequence Requirements', fontsize=20)
    plt.savefig(os.path.join(OUTPUT_DIR, "4_solution_comparison.png"), dpi=300)
    plt.close(fig)
    
    # Clean up temp files
    os.remove(solution_2_gears_path)
    os.remove(solution_4_gears_path)
    print("Visual 4 saved.")

# --- Main Execution ---
if __name__ == "__main__":
    generate_problem_definition_visual()
    generate_analysis_pipeline_visual()
    generate_training_graph_visual()
    generate_comparison_visual()
    print("\nAll visuals have been generated in the 'thesis_visuals' folder.")

###################### Justification