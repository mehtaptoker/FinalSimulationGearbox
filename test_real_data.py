import sys
sys.path.append(".")

from pathfinding.finder import Pathfinder

finder = Pathfinder()
path = finder.find_path("data/intermediate/Example1_processed.json")

if path:
    print(f"Path found with {len(path)} points")
    print("First 5 points:", path[:5])
else:
    print("No path found")
