import numba as nb
import numpy as np
import pygame
import matplotlib.pyplot as plt
import math
import os
import json
import pickle
np.random.seed(None)

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

def add_newboid(parent, boids, classes, energies, boid_ids, next_boid_id, param_dict, params=None):
    new_row = np.array(
        [np.random.uniform(0, WIDTH), np.random.uniform(0, HEIGHT), np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
        dtype=np.float32)
    new_row = new_row.reshape(1, -1)
    boids = np.append(boids, new_row, axis=0)
    classes = np.append(classes, classes[parent])
    if classes[parent] == 1:
        energies = np.append(energies, ENERGY_TO_REPRODUCE)
    else:
        energies = np.append(energies, MAX_ENERGY[classes[parent]])

    if params is not None:
        # Recreate global params, for testing
        # boid_type = classes[parent]
        # boid_class = np.eye(len(CLASSES), dtype=int)[boid_type]
        # new_params = create_params_array(boid_class, SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT)
        # params = np.append(params, new_params, axis=0)
        # Compleatly random
        # new_params = np.random.uniform(0, 1, size=(1, 3, len(CLASSES)))
        # Mutation
        parent_params = params[parent]
        new_params = np.random.normal(loc=parent_params, scale=0.1, size=parent_params.shape)
        params = np.append(params, new_params[None, ...], axis=0)

        # Track new boid's parameters
        param_dict[next_boid_id] = new_params

    # Assign unique ID to the new boid
    next_boid_id += 1
    boid_ids = np.append(boid_ids, next_boid_id)

    return boids, classes, energies, boid_ids, next_boid_id, params

def remove_boid(boid, boids, classes, energies, random_factors, boid_ids, params=None):
    if params is not None:
        params = np.delete(params, boid, axis=0) 
    return np.delete(boids, boid, 0), np.delete(classes, boid), np.delete(energies, boid), np.delete(random_factors, boid), np.delete(boid_ids, boid), params

@nb.njit
def frobenius_norm(a):
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
    # Turn around a screen edges
    boids[:, 2] = boids[:, 2] + (boids[:, 0] < MARGIN_LEFT) * TURN_FACTOR
    boids[:, 2] = boids[:, 2] - (boids[:, 0] > MARGIN_RIGHT) * TURN_FACTOR
    boids[:, 3] = boids[:, 3] - (boids[:, 1] > MARGIN_BOTTOM) * TURN_FACTOR
    boids[:, 3] = boids[:, 3] + (boids[:, 1] < MARGIN_TOP) * TURN_FACTOR

    mask = np.logical_or(np.logical_or(np.logical_or(boids[:, 0] < 0, boids[:, 0] > WIDTH), boids[:, 1] < 0), boids[:, 1] > HEIGHT)
    boids[:, 2] = boids[:, 2] * np.logical_not(mask) - boids[:, 2] * mask
    boids[:, 3] = boids[:, 3] * np.logical_not(mask) - boids[:, 3] * mask

@nb.njit
def get_perents(classes, random_factors, energies, gametic):
    parents = None
    for c in range(len(CLASSES)):
        class_mask = classes == c
        birth_mask = random_factors == (gametic % max(random_factors))
        if c == 1:
            birth_mask = birth_mask * (energies > ENERGY_TO_REPRODUCE)
        if parents is None:
            parents = class_mask * birth_mask
        else:
            extra_parents = class_mask * birth_mask
            parents = np.logical_or(parents, extra_parents)
    return np.nonzero(parents)[0]

@nb.njit
def update_numba(boids, classes, energies, random_factors, gametic, params=None):
    deleteable_boids = [0]
    resetEnergie = [0]
    deleteable_boids.pop()
    resetEnergie.pop()

    for i in range(len(boids)):
        if energies[i] <= 0:
            deleteable_boids.append(i)
            continue
        
        # should do the same thing but does not
        if params is None:
            separation_weight = SEPARATION_WEIGHT[classes[i]]
            alignment_weight = ALIGNMENT_WEIGHT[classes[i]]
            cohesion_weight = COHESION_WEIGHT[classes[i]]
        else:
            separation_weight = params[i, 0, :]
            alignment_weight = params[i, 1, :]
            cohesion_weight = params[i, 2, :]

        angle_mask = create_angle_mask(boids, i, angle=3)
        distances = frobenius_norm(boids[i, :2] - boids[:, :2])

        for c in range(len(CLASSES)):
            class_mask = classes == c
            class_mask = np.stack((class_mask, class_mask), axis=1)

            # Use makes to find neighbors
            sep_mask = distances < SEPARATION_RADIUS[classes[i], c]
            sep_mask = np.logical_and(sep_mask, angle_mask)
            sep_mask = np.stack((sep_mask, sep_mask), axis=1)
            visible_mask = np.logical_and(SEPARATION_RADIUS[classes[i], c] <= distances, distances < VISIBLE_RADIUS[classes[i], c], angle_mask)
            visible_mask = np.stack((visible_mask, visible_mask), axis=1)
            
            num_neighbors = np.sum(visible_mask*class_mask)/2

            # Separation
            boids[i, 2:] = boids[i, 2:] + np.sum((boids[i, :2] - boids[:, :2])*sep_mask*class_mask, axis=0) * separation_weight[c]
            
            # Check if neighboring_boids>0
            if num_neighbors > 0:
                # Alignment
                boids[i, 2:] = boids[i, 2:] + (np.sum((boids[:, 2:])*visible_mask*class_mask, axis=0) / num_neighbors - boids[i, 2:]) * alignment_weight[c]

                # Cohesion
                boids[i, 2:] = boids[i, 2:] + (np.sum((boids[:, :2])*visible_mask*class_mask, axis=0) / num_neighbors - boids[i, :2]) * cohesion_weight[c]

        check_collisions(i, classes, distances, resetEnergie, deleteable_boids)

    without_duplicates = list(set(deleteable_boids))

    turn_at_egdes(boids)

    # Normalize speed
    speed = np.sqrt(boids[:, 2]**2 + boids[:, 3]**2)[:, np.newaxis]
    scale = MAX_SPEED[classes][:, np.newaxis] / speed
    boids[:, 2:] = boids[:, 2:] * scale

    # Update
    boids[:, :2] = boids[:, :2] + boids[:, 2:]

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

# Function to create the combined weight array
def create_params_array(classes, separation_weight, alignment_weight, cohesion_weight):
    separation_result = np.vstack([separation_weight[i] for i, count in enumerate(classes) for _ in range(count)])
    alignment_result = np.vstack([alignment_weight[i] for i, count in enumerate(classes) for _ in range(count)])
    cohesion_result = np.vstack([cohesion_weight[i] for i, count in enumerate(classes) for _ in range(count)])
    
    combined_result = np.stack((separation_result, alignment_result, cohesion_result), axis=1)
    return combined_result

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
    
    # To use global params
    # params = None 
    # To test if local params work the same as global
    params = create_params_array(CLASSES, SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT) 
    # Initolize params randomly
    # params = np.random.uniform(0, 1, size=(NUM_BOIDS, 3, len(CLASSES))) 

    classes = np.concatenate([[i] * number for i, number in enumerate(CLASSES)])
    # energies = np.concatenate([[MAX_ENERGY[i]] * number for i, number in enumerate(CLASSES)])
    energies = np.concatenate([[MAX_ENERGY[0]] * CLASSES[0], [ENERGY_TO_REPRODUCE] * CLASSES[1]])
    random_factors = np.concatenate([np.random.randint(1, REPRODUCE_CYCLE[i], size=number) for i, number in enumerate(CLASSES)]) 

    # Initialize unique IDs for each boid
    boid_ids = np.arange(NUM_BOIDS)
    next_boid_id = NUM_BOIDS

    family_tree = []
    param_dict = {boid_id: params[i] for i, boid_id in enumerate(boid_ids)}

    running = True
    gametic = 0

    boid_counts = []

    while running and len(boids) != 0:
        screen.fill(BACKGROUND_COLOR)
        draw_dotted_margin(screen, WIDTH, HEIGHT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        deleteableBoids, energiesToReset, parents = update_numba(boids, classes, energies, random_factors, gametic, params=params)
        for boid in energiesToReset:
            energies[boid] = MAX_ENERGY[classes[boid]]
        
        if len(parents) != 0:
            for parent in parents:
                # If the perent is a pray and the total number of prays has exceded five time the original amound, 
                # we assume that the food limit for the prey is reached, and we don't add the new pray.
                if not (classes[parent] == 0 and np.sum(classes == 0) > CLASSES[0]*6):
                    family_tree.append((boid_ids[parent], next_boid_id))
                    boids, classes, energies, boid_ids, next_boid_id, params = add_newboid(parent, boids, classes, energies, boid_ids, next_boid_id, param_dict, params=params)
                    random_factors = np.append(random_factors, np.random.randint(1, REPRODUCE_CYCLE[classes[parent]], 1))

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

    return boid_counts, family_tree, param_dict

def plot_boid_counts(boid_counts, num_classes):
    class_names = ['Prey', 'Predator']

    if not os.path.exists("figures"):
        os.makedirs("figures")

    for class_idx in range(num_classes):
        class_counts = [counts[class_idx] for counts in boid_counts]
        plt.plot(class_counts, label=f'{class_names[class_idx]}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Number of Boids')
    plt.title('Number of Boids per Class Over Time')
    plt.legend()
    plt.savefig('figures/boid_counts.png')
    plt.close()

def plot_family_tree(param_dict, family_tree, param_index_pairs):
    fig, axes = plt.subplots(len(param_index_pairs), figsize=(12, 8))
    
    if len(param_index_pairs) == 1:
        axes = [axes]

    for ax, (param_x, param_y) in zip(axes, param_index_pairs):
        print(param_x, param_y)
        # Plot the parameters for all boids
        for boid_id, params in param_dict.items():
            print(params)
            break
            # for class_idx in range(params.shape[1]):
    #             ax.scatter(params[param_x, class_idx], params[param_y, class_idx], label=f'Class {class_idx}')
        
    #     # Draw parent-child lines
    #     for parent_id, child_id in family_tree:
    #         parent_params = param_dict[parent_id]
    #         child_params = param_dict[child_id]
    #         ax.plot([parent_params[param_x], child_params[param_x]], 
    #                 [parent_params[param_y], child_params[param_y]], 'k-')
        
    #     ax.set_xlabel(f'Parameter {param_x}')
    #     ax.set_ylabel(f'Parameter {param_y}')
    #     ax.legend()
    
    plt.tight_layout()
    plt.show()

def save_simulation_data(filename, boid_counts, family_tree, param_dict):
    with open(filename, 'wb') as f:
        pickle.dump((boid_counts, family_tree, param_dict), f)

def load_simulation_data(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return None

if __name__ == "__main__":
    filename = 'simulation_data.pkl'
    load_data = True  # Set this to False to run the simulation instead of loading data
    visual = True # if True uses pygame to visualize the boids simulation
    data = load_simulation_data(filename)

    if load_data and data is not None:
        boid_counts, family_tree, param_dict = data
    else:
        boid_counts, family_tree, param_dict = simulation(visual, sim_length=None) # Set sim_length to the number of simulation steps you want to run
        save_simulation_data(filename, boid_counts, family_tree, param_dict)

    plot_family_tree(param_dict, family_tree, [(0, 1), (0, 2), (1, 2)])
    plot_boid_counts(boid_counts, len(CLASSES))


