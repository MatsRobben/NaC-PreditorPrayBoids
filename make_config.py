import json
import os

def create_config(filename, config_data):
    with open(filename, 'w') as file:
        json.dump(config_data, file, indent=4)

# Example config data
config1 = {
    "WIDTH": 800,
    "HEIGHT": 800,
    "CLASSES": [50, 10],
    "NUM_CLASSES": 60,
    "VISIBLE_RADIUS": [[50, 50], [50, 50]],
    "SEPARATION_RADIUS": [[10, 45], [10, 20]],
    "ALIGNMENT_WEIGHT": [[0.05, 0.05], [0.05, 0.1]],
    "COHESION_WEIGHT": [[0.05, 0.05], [0.05, 0.05]],
    "SEPARATION_WEIGHT": [[0.8, 1], [0, 0.1]],
    "MAX_ENERGY": [700, 800],
    "ENERGY_TO_REPRODUCE": 700,
    "ENERGY_EATING": 100,
    "REPRODUCE_CYCLE": [200, 400],
    "DISTANCE_TO_EAT": 10,
    "PARAM_DEVIATION": 0.1,
    "TURN_FACTOR": 0.2,
    "BOID_LENGTH": [10, 14],
    "BACKGROUND_COLOR": [220, 220, 220],
    "BOID_COLOR": [[0, 0, 0], [255, 0, 0]],
    "MAX_SPEED": [3.5, 4],
    "MARGIN_LEFT": 100,
    "MARGIN_RIGHT": 700,
    "MARGIN_TOP": 100,
    "MARGIN_BOTTOM": 700
}

if __name__ == "__main__":
    os.makedirs('configs', exist_ok=True)
    create_config('configs/config1.json', config1)

