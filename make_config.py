import json
import os

def create_config(filename, config_data):
    with open(filename, 'w') as file:
        json.dump(config_data, file, indent=4)

def load_config(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def update_config(loaded_config, default_config):
    updated_config = loaded_config.copy()
    
    for key, default_value in default_config.items():
        if key not in loaded_config:
            updated_value = input(f"Variable '{key}' is missing. Enter a value (default {default_value}): ")
            updated_config[key] = type(default_value)(json.loads(updated_value) if updated_value else default_value)
        elif type(loaded_config[key]) != type(default_value):
            updated_value = input(f"Variable '{key}' has a different type. Enter a value (default {default_value}): ")
            updated_config[key] = type(default_value)(json.loads(updated_value) if updated_value else default_value)
    
    return updated_config

# Example default config data
default_config = {
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
    make_config = False
    update_existing = True

    if make_config:
        os.makedirs('configs', exist_ok=True)
        create_config('configs/config1.json', default_config)

    if update_existing:
        # Load the existing config
        filename = 'configs/config_cohesive_flocking.json'
        loaded_config = load_config(filename)
        
        # Update the config with user inputs for missing or mismatched variables
        updated_config = update_config(loaded_config, default_config)
        
        # Save the updated config back to the file
        create_config(filename, updated_config)

        print("Config file updated successfully!")
