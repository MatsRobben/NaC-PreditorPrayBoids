import numba as nb
import numpy as np
import pygame
import math
import os
import json
import pickle
import figures
np.random.seed(0)

def load_config(config_name):
    with open(f'configs/{config_name}.json', 'r') as file:
        config = json.load(file)
    return config

# Example of loading a config
config_name = 'config_cohesive_flocking'  # Change this to load different configs
config = load_config(config_name)

# Assign individual variables
WIDTH = config['WIDTH']
HEIGHT = config['HEIGHT']
CLASSES = np.array(config['CLASSES'])
NUM_BOIDS = np.sum(CLASSES)
VISIBLE_RADIUS = np.array(config['VISIBLE_RADIUS'])
SEPARATION_RADIUS = np.array(config['SEPARATION_RADIUS'])
ALIGNMENT_WEIGHT = np.array(config['ALIGNMENT_WEIGHT'])
COHESION_WEIGHT = np.array(config['COHESION_WEIGHT'])
SEPARATION_WEIGHT = np.array(config['SEPARATION_WEIGHT'])
MAX_ENERGY = np.array(config['MAX_ENERGY'])
ENERGY_TO_REPRODUCE = config['ENERGY_TO_REPRODUCE']
ENERGY_EATING = config['ENERGY_EATING']
REPRODUCE_CYCLE = np.array(config['REPRODUCE_CYCLE'])
DISTANCE_TO_EAT = config['DISTANCE_TO_EAT']
PARAM_DEVIATION = config['PARAM_DEVIATION']
TURN_FACTOR = config['TURN_FACTOR']
BOID_LENGTH = config['BOID_LENGTH']
BACKGROUND_COLOR = config['BACKGROUND_COLOR']
BOID_COLOR = np.array(config['BOID_COLOR'])
MAX_SPEED = np.array(config['MAX_SPEED'])
MARGIN_LEFT = config['MARGIN_LEFT']
MARGIN_RIGHT = config['MARGIN_RIGHT']
MARGIN_TOP = config['MARGIN_TOP']
MARGIN_BOTTOM = config['MARGIN_BOTTOM']

EVOLUTION = True

def add_newboid(parent, boids, classes, energies, boid_ids, next_boid_id, param_dict, params=None):
    # Add new boid.
    # Randomly assign boid a posistion and velocity vector
    new_row = np.array(
        [np.random.uniform(0, WIDTH), np.random.uniform(0, HEIGHT), np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
        dtype=np.float32)
    new_row = new_row.reshape(1, -1)
    boids = np.append(boids, new_row, axis=0)

    # Give new boids the parents class
    classes = np.append(classes, classes[parent])

    # Preditor does not get full energy when born
    if classes[parent] == 1:
        energies = np.append(energies, ENERGY_TO_REPRODUCE)
    else:
        energies = np.append(energies, MAX_ENERGY[classes[parent]])

    # Create new parameters
    if EVOLUTION:
        # Mutation
        parent_params = params[parent]
        new_params = np.random.normal(loc=parent_params, scale=PARAM_DEVIATION, size=parent_params.shape)
        params = np.append(params, new_params[None, ...], axis=0)
    else:
        # Recreate global params
        boid_type = classes[parent]
        boid_class = np.eye(len(CLASSES), dtype=int)[boid_type]
        new_params = create_params_from_single_point(boid_class, [SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT])
        params = np.append(params, new_params, axis=0)
        
    # Track new boid's parameters
    param_dict[next_boid_id] = (new_params, classes[parent])

    # Assign unique ID to the new boid
    boid_ids = np.append(boid_ids, next_boid_id)
    next_boid_id += 1

    return boids, classes, energies, boid_ids, next_boid_id, params

def remove_boid(boid_idx, boids, classes, energies, random_factors, boid_ids, params=None):
    # Delete single boids (boid_idx) for all of the arrays.
    return np.delete(boids, boid_idx, 0), \
           np.delete(classes, boid_idx), \
           np.delete(energies, boid_idx), \
           np.delete(random_factors, boid_idx), \
           np.delete(boid_ids, boid_idx), \
           np.delete(params, boid_idx, axis=0) 

@nb.njit
def frobenius_norm(a):
    # Compute Euclidean distances between points
    norms = np.empty(a.shape[0], dtype=a.dtype)
    for i in nb.prange(a.shape[0]):
        norms[i] = np.sqrt(a[i, 0] * a[i, 0] + a[i, 1] * a[i, 1])
    return norms

@nb.njit
def check_collisions(current_boid, classes, distances, resetEnergie, deleteable_boids):
    if classes[current_boid] == 1:  # Class 0 checking for collisions with Class 1
            collision_mask = (classes == 0) & (distances < DISTANCE_TO_EAT)
            if np.any(collision_mask):
                resetEnergie.append(current_boid)
                deleteable_boids.append(np.where(collision_mask)[0][0])

@nb.njit
def create_angle_mask(boids, current_boid, angle=3):
    # Calculate the angle between the current boid and the neighbor
    angle_to_neighbor = np.arctan2(boids[:, 1] - boids[current_boid, 1], boids[:, 0] - boids[current_boid, 0])
    # Calculate the angle difference between the current boid's velocity direction and the angle to the neighbor
    angle_difference = np.abs(np.arctan2(boids[current_boid, 3], boids[current_boid, 2]) - angle_to_neighbor)
    # 1 = 90 degrees
    # 2 = 180 degrees
    # 3 = 270 degrees
    # 4 = 360 degrees
    vision_angle = math.pi * angle
    # Check if the neighbor is within the vision angle range
    return angle_difference <= vision_angle / 2

@nb.njit
def turn_at_egdes(boids):
    # Stear away from the screen edges when the boid passes the margins.
    boids[:, 2] = boids[:, 2] + (boids[:, 0] < MARGIN_LEFT) * TURN_FACTOR
    boids[:, 2] = boids[:, 2] - (boids[:, 0] > MARGIN_RIGHT) * TURN_FACTOR
    boids[:, 3] = boids[:, 3] - (boids[:, 1] > MARGIN_BOTTOM) * TURN_FACTOR
    boids[:, 3] = boids[:, 3] + (boids[:, 1] < MARGIN_TOP) * TURN_FACTOR

    # Turn around 180 degrees when the boid hits the real border.
    mask = np.logical_or(np.logical_or(np.logical_or(boids[:, 0] < 0, boids[:, 0] > WIDTH), boids[:, 1] < 0), boids[:, 1] > HEIGHT)
    boids[:, 2] = boids[:, 2] * np.logical_not(mask) - boids[:, 2] * mask
    boids[:, 3] = boids[:, 3] * np.logical_not(mask) - boids[:, 3] * mask

@nb.njit
def get_perents(classes, random_factors, energies, gametic):
    # Compute new parents based on energies and the random factors.

    parents = None
    for c in range(len(CLASSES)):
        class_mask = classes == c

        # We use the % and the gametics to loop over random_factors, 
        # so every boid has the chance to be a parent every X iterations
        birth_mask = random_factors == (gametic % max(random_factors))

        # If the parent is a preditor we also want to check if the energy is sufficient
        if c == 1:
            birth_mask = birth_mask * (energies > ENERGY_TO_REPRODUCE)
        
        # Add to parents
        if parents is None:
            parents = class_mask * birth_mask
        else:
            extra_parents = class_mask * birth_mask
            parents = np.logical_or(parents, extra_parents)
    return np.nonzero(parents)[0]

@nb.njit
def update_boids(boids, classes, energies, random_factors, gametic, params=None):
    # Because of numba we have to fist add an empty element before removing it 
    deleteable_boids = [0]
    resetEnergie = [0]
    deleteable_boids.pop()
    resetEnergie.pop()

    # Loop over boids and compute their movement based on their neighbors
    for i in range(len(boids)):

        # Boids die when they have no energy left
        if energies[i] <= 0:
            deleteable_boids.append(i)
            continue
        
        # Select the parameters of boid
        separation_weight = params[i, 0, :]
        alignment_weight = params[i, 1, :]
        cohesion_weight = params[i, 2, :]

        # Compute the neighboring boids and the angle which the boid can see
        angle_mask = create_angle_mask(boids, i, angle=3)
        distances = frobenius_norm(boids[i, :2] - boids[:, :2])

        # Compute movement individualy for the classes
        for c in range(len(CLASSES)):
            # Mask of the selected class
            class_mask = classes == c
            class_mask = np.stack((class_mask, class_mask), axis=1)

            # Use masks to find neighbors in seperation range and visible range
            sep_mask = distances < SEPARATION_RADIUS[classes[i], c]
            sep_mask = np.logical_and(sep_mask, angle_mask)
            sep_mask = np.stack((sep_mask, sep_mask), axis=1)
            visible_mask = np.logical_and(SEPARATION_RADIUS[classes[i], c] <= distances, distances < VISIBLE_RADIUS[classes[i], c], angle_mask)
            visible_mask = np.stack((visible_mask, visible_mask), axis=1)
            
            # Compute number of neighbors, devided by 2 because of the stacking (needed for multiplication without broadcasting).
            num_neighbors = np.sum(visible_mask*class_mask)/2

            # Separation
            boids[i, 2:] = boids[i, 2:] + np.sum((boids[i, :2] - boids[:, :2])*sep_mask*class_mask, axis=0) * separation_weight[c]
            
            # Check if neighboring_boids>0
            if num_neighbors > 0:
                # Alignment
                boids[i, 2:] = boids[i, 2:] + (np.sum((boids[:, 2:])*visible_mask*class_mask, axis=0) / num_neighbors - boids[i, 2:]) * alignment_weight[c]

                # Cohesion
                boids[i, 2:] = boids[i, 2:] + (np.sum((boids[:, :2])*visible_mask*class_mask, axis=0) / num_neighbors - boids[i, :2]) * cohesion_weight[c]

        # Check if the predators have eaten any of the prey
        check_collisions(i, classes, distances, resetEnergie, deleteable_boids)

    # Select the set of boids that have died
    without_duplicates = list(set(deleteable_boids))

    # Function to make sure the boids stay in the given window
    turn_at_egdes(boids)

    # Normalize speed
    speed = np.sqrt(boids[:, 2]**2 + boids[:, 3]**2)[:, np.newaxis]
    scale = MAX_SPEED[classes][:, np.newaxis] / speed
    boids[:, 2:] = boids[:, 2:] * scale

    # Update, add the velocity vector to the possition vector. 
    boids[:, :2] = boids[:, :2] + boids[:, 2:]

    # Decides which Boids will reproduce this iteration
    parents = get_perents(classes, random_factors, energies, gametic)

    return without_duplicates, resetEnergie, parents

def draw_boids(screen, flock, classes):
    global BOID_LENGTH

    for i, boid in enumerate(flock):
        c = classes[i]

        angle = math.atan2(boid[3], boid[2])
        p1 = (boid[0] + BOID_LENGTH[c] * math.cos(angle), boid[1] + BOID_LENGTH[c] * math.sin(angle))
        p2 = (boid[0] + BOID_LENGTH[c] * math.cos(angle - 2.5), boid[1] + BOID_LENGTH[c] * math.sin(angle - 2.5))
        p3 = (boid[0] + BOID_LENGTH[c] * math.cos(angle + 2.5), boid[1] + BOID_LENGTH[c] * math.sin(angle + 2.5))
        pygame.draw.polygon(screen, BOID_COLOR[c], (p1, p2, p3))

def draw_dotted_margin(screen, width, height, dot_length=10, dot_spacing=5, color=(255, 255, 255), thickness=2):
    global MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM

    # Draw the margin lines
    line_color = color
    line_thickness = thickness

    # Left margin
    for y in range(0, height, dot_length + dot_spacing * 2):
        pygame.draw.line(screen, line_color, (MARGIN_LEFT, y), (MARGIN_LEFT, min(y + dot_length, height)), line_thickness)
    
    # Right margin
    for y in range(0, height, dot_length + dot_spacing * 2):
        pygame.draw.line(screen, line_color, (MARGIN_RIGHT, y), (MARGIN_RIGHT, min(y + dot_length, height)), line_thickness)
    
    # Top margin
    for x in range(0, width, dot_length + dot_spacing * 2):
        pygame.draw.line(screen, line_color, (x, MARGIN_TOP), (min(x + dot_length, width), MARGIN_TOP), line_thickness)
    
    # Bottom margin
    for x in range(0, width, dot_length + dot_spacing * 2):
        pygame.draw.line(screen, line_color, (x, MARGIN_BOTTOM), (min(x + dot_length, width), MARGIN_BOTTOM), line_thickness)

def create_params_from_single_point(classes, array):
    # Function to create parameter arrays from the global parameters.
    results = []
    for weights in array:
        result = np.vstack([[weights[i]] * count for i, count in enumerate(classes)])
        results.append(result)

    combined_result = np.stack(results, axis=1)
    return combined_result

def create_params_from_ranges(ranges):
    # Randomly initiolize the parameters between given ranges.
    # These can be different for each individual parameter. 
    params = np.empty((NUM_BOIDS, 3, 2))

    start_idx = 0
    for class_idx, num_boids in enumerate(CLASSES):
        end_idx = start_idx + num_boids
        for param_idx in range(3):
            min_self_val, max_self_val = ranges[class_idx][param_idx][0]
            min_other_val, max_other_val = ranges[class_idx][param_idx][1]

            params[start_idx:end_idx, param_idx, 0] = np.random.uniform(min_self_val, max_self_val, num_boids)
            params[start_idx:end_idx, param_idx, 1] = np.random.uniform(min_other_val, max_other_val, num_boids)

        start_idx = end_idx
    return params

def add_quantiles(params, quantiles_prey, quantiles_predator, classes):
    # Extract 'Prey' and 'Predator' parameters
    prey_params = params[classes == 0]
    predator_params = params[classes == 1]

    # Compute quantiles for 'Prey'
    quantiles_prey['q1'].append(np.quantile(prey_params, 0.01, axis=0))
    quantiles_prey['q5'].append(np.quantile(prey_params, 0.05, axis=0))
    quantiles_prey['q25'].append(np.quantile(prey_params, 0.25, axis=0))
    quantiles_prey['q50'].append(np.median(prey_params, axis=0))
    quantiles_prey['q75'].append(np.quantile(prey_params, 0.75, axis=0))
    quantiles_prey['q95'].append(np.quantile(prey_params, 0.95, axis=0))
    quantiles_prey['q99'].append(np.quantile(prey_params, 0.99, axis=0))

    # Compute quantiles for 'Predator'
    quantiles_predator['q1'].append(np.quantile(predator_params, 0.01, axis=0))
    quantiles_predator['q5'].append(np.quantile(predator_params, 0.05, axis=0))
    quantiles_predator['q25'].append(np.quantile(predator_params, 0.25, axis=0))
    quantiles_predator['q50'].append(np.median(predator_params, axis=0))
    quantiles_predator['q75'].append(np.quantile(predator_params, 0.75, axis=0))
    quantiles_predator['q95'].append(np.quantile(predator_params, 0.95, axis=0))
    quantiles_predator['q99'].append(np.quantile(predator_params, 0.99, axis=0))

def simulation(visual=True, sim_length=None):
    if visual:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Boids Simulation")
        clock = pygame.time.Clock()

    boids = np.array([np.random.uniform(0, WIDTH, size=NUM_BOIDS),  # x coordiante
                  np.random.uniform(0, HEIGHT, size=NUM_BOIDS), # y coordiante
                  np.random.uniform(-1, 1, size=NUM_BOIDS),     # xv velocity vector in x direction
                  np.random.uniform(-1, 1, size=NUM_BOIDS),      # yv velocity vector in y direction
                  ], dtype=np.float32).T  
    
    if EVOLUTION:
        ranges = [
            [[(0.0, 0.5), (0.8, 1.0)], [(0.2, 0.6), (0.4, 0.8)], [(0.0, 0.3), (0.5, 0.9)]], # Prey
            [[(0.1, 0.5), (0.3, 0.7)], [(0.2, 0.6), (0.4, 0.8)], [(0.0, 0.3), (0.5, 0.9)]]  # Predators
        ]
        params = create_params_from_ranges(ranges)
    else:
        # Global parameters
        params = create_params_from_single_point(classes, [SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT]) 
    
    # Arrays for Boid classes energies and random factors
    classes = np.concatenate([[i] * number for i, number in enumerate(CLASSES)])
    energies = np.concatenate([[MAX_ENERGY[0]] * CLASSES[0], [ENERGY_TO_REPRODUCE] * CLASSES[1]])
    random_factors = np.concatenate([np.random.randint(1, REPRODUCE_CYCLE[i], size=number) for i, number in enumerate(CLASSES)]) 

    # Initialize unique IDs for each boid
    boid_ids = np.arange(NUM_BOIDS)
    next_boid_id = NUM_BOIDS

    # Save information for making plots
    family_tree = []
    param_dict = {boid_id: (params[i], classes[i]) for i, boid_id in enumerate(boid_ids)}
    quantiles_prey = {q: [] for q in ['q1', 'q5', 'q25', 'q50', 'q75', 'q95', 'q99']}
    quantiles_predator = {q: [] for q in ['q1', 'q5', 'q25', 'q50', 'q75', 'q95', 'q99']}
    boid_counts = []

    running = True
    gametic = 0

    # Run while not exited pygame and non of the classes are empty
    while running and np.sum(classes == 0) != 0 and np.sum(classes == 1) != 0:
        if visual:
            screen.fill(BACKGROUND_COLOR)
            draw_dotted_margin(screen, WIDTH, HEIGHT)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # Compute quantiles of parameters
        add_quantiles(params, quantiles_prey, quantiles_predator, classes)

        # Run main update loop
        deleteableBoids, energiesToReset, parents = update_boids(boids, classes, energies, random_factors, gametic, params=params)
        
        # Add energy for those boids that have eaten
        for boid in energiesToReset:
            energies[boid] = max(ENERGY_EATING + energies[boid], MAX_ENERGY[1])
        
        # Add children
        if len(parents) != 0:
            for parent in parents:
                # If the perent is a prey and the total number of preys has exceded five time the original amound, 
                # we assume that the food limit for the prey is reached, and we don't add the new prey.
                if not (classes[parent] == 0 and np.sum(classes == 0) > CLASSES[0]*6):
                    family_tree.append((boid_ids[parent], next_boid_id))
                    boids, classes, energies, boid_ids, next_boid_id, params = add_newboid(parent, boids, classes, energies, boid_ids, next_boid_id, param_dict, params=params)
                    random_factors = np.append(random_factors, np.random.randint(1, REPRODUCE_CYCLE[classes[parent]], 1))

        # Remove boids that have been eaten or died of starvation
        boids, classes, energies, random_factors, boid_ids, params = remove_boid(deleteableBoids, boids, classes, energies, random_factors, boid_ids, params=params)

        if visual:
            draw_boids(screen, boids, classes)

        # Count the number of boids per class
        counts = [np.sum(classes == i) for i in range(len(CLASSES))]
        boid_counts.append(counts)

        if visual:
            pygame.display.flip()
            clock.tick(30)

        gametic += 1
        energies -= 1

    if visual:
        pygame.quit()

    return boid_counts, family_tree, param_dict, quantiles_prey, quantiles_predator

def save_simulation_data(filename, *args):
    # Save simulation variables, so that plots can be remade without having to rerun the simulation.
    data_to_dump = args
    
    with open(filename, 'wb') as f:
        pickle.dump(data_to_dump, f)

def load_simulation_data(filename):
    # Load simulation variables, so that plots can be remade without having to rerun the simulation.
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return None

if __name__ == "__main__":
    filename = 'simulation_data.pkl'
    load_data = False  # Set this to False to run the simulation instead of loading data
    visual = True # if True uses pygame to visualize the boids simulation
    data = load_simulation_data(filename)

    if load_data and data is not None:
        boid_counts, family_tree, param_dict, quantiles_prey, quantiles_predator = data
    else:
        boid_counts, family_tree, param_dict, quantiles_prey, quantiles_predator = simulation(visual, sim_length=None) # Set sim_length to the number of simulation steps you want to run
        save_simulation_data(filename, boid_counts, family_tree, param_dict, quantiles_prey, quantiles_predator)

    figures.plot_family_tree(param_dict, family_tree, config_name)
    figures.plot_boid_counts(boid_counts, len(CLASSES), config_name)
    figures.plot_distribution_over_time(quantiles_prey, quantiles_predator, config_name)


