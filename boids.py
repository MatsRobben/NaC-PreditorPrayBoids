import numba as nb
import numpy as np
import pygame
import matplotlib.pyplot as plt
import math
import os
np.random.seed(None)

WIDTH, HEIGHT = 800, 800

CLASSES = np.array([100, 10])
NUM_BOIDS = np.sum(CLASSES)
VISIBLE_RADIUS = np.array([[50, 50],
                          [50, 50]])
SEPARATION_RADIUS = np.array([[10, 45],
                             [10, 20]])
ALIGNMENT_WEIGHT = np.array([[0.05, 0.05], 
                             [0.05, 0.1]])
COHESION_WEIGHT = np.array([[0.005, 0.005], 
                            [0.005, 0.005]])
SEPARATION_WEIGHT = np.array([[0.1, 0.9], 
                              [0.1, 0.1]])

# adding timers per game tick
MAX_ENERGY = np.array([300, 400])
ENERGY_TO_REPRODUCE = 300
REPRODUCE_CYCLE = 200
DISTANCE_TO_EAT = 10

TRUN_FACTOR = 0.2
BOID_LENGTH = [10, 14]
BACKGROUND_COLOR = (220, 220, 220)
BOID_COLOR = [(0, 0, 0), (255, 0, 0)]

MAX_SPEED = np.array([3, 5])
MARGIN_LEFT=100; MARGIN_RIGHT=WIDTH-100; MARGIN_TOP=100; MARGIN_BOTTOM=HEIGHT-100

@nb.njit
def frobenius_norm(a):
    norms = np.empty(a.shape[0], dtype=a.dtype)
    for i in nb.prange(a.shape[0]):
        norms[i] = np.sqrt(a[i, 0] * a[i, 0] + a[i, 1] * a[i, 1])
    return norms

def add_newboid(parent, boids, classes, energies, params=None):
    new_row = np.array(
        [np.random.uniform(0, WIDTH), np.random.uniform(0, HEIGHT), np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
        dtype=np.float32)
    new_row = new_row.reshape(1, -1)
    boids = np.append(boids, new_row, axis=0)
    classes = np.append(classes, classes[parent])
    energies = np.append(energies, MAX_ENERGY[classes[parent]])
    if params is not None:
        # Recreate global params, for testing
        # boid_class = np.eye(len(CLASSES), dtype=int)[boid_type]
        # new_params = create_params_array(boid_class, SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT)
        # Compleatly random
        # new_params = np.random.uniform(0, 1, size=(1, 3, len(CLASSES)))
        # Mutation
        parent_params = params[parent]
        new_params = np.random.normal(loc=parent_params, scale=0.1, size=parent_params.shape)
        params = np.append(params, new_params[None, ...], axis=0)

    return boids, classes, energies, params

def remove_boid(boid, boids, classes, energies, random_factors, params=None):
    if params is not None:
        params = np.delete(params, boid, axis=0) 
    return np.delete(boids, boid, 0), np.delete(classes, boid), np.delete(energies, boid), np.delete(random_factors, boid), params

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
    boids[:, 2] = boids[:, 2] + (boids[:, 0] < MARGIN_LEFT) * TRUN_FACTOR
    boids[:, 2] = boids[:, 2] - (boids[:, 0] > MARGIN_RIGHT) * TRUN_FACTOR
    boids[:, 3] = boids[:, 3] - (boids[:, 1] > MARGIN_BOTTOM) * TRUN_FACTOR
    boids[:, 3] = boids[:, 3] + (boids[:, 1] < MARGIN_TOP) * TRUN_FACTOR

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

def pygame_sim():
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
    # params = create_params_array(CLASSES, SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT) 
    # Initolize params randomly
    params = np.random.uniform(0, 1, size=(NUM_BOIDS, 3, len(CLASSES))) 

    classes = np.concatenate([[i] * number for i, number in enumerate(CLASSES)])
    energies = np.concatenate([[MAX_ENERGY[i]] * number for i, number in enumerate(CLASSES)])

    random_factors = np.random.randint(1, REPRODUCE_CYCLE, size=classes.shape)

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
                boids, classes, energies, params = add_newboid(parent, boids, classes, energies, params=params)
                random_factors = np.append(random_factors, np.random.randint(1, REPRODUCE_CYCLE, 1))

        boids, classes, energies, random_factors, params = remove_boid(deleteableBoids, boids, classes, energies, random_factors, params=params)

        draw_boids(screen, boids, classes)

        # Count the number of boids per class
        counts = [np.sum(classes == i) for i in range(len(CLASSES))]
        boid_counts.append(counts)

        pygame.display.flip()
        clock.tick(30)
        gametic += 1
        energies -= 1

    pygame.quit()

    return boid_counts

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

if __name__ == "__main__":
    boid_counts = pygame_sim()
    plot_boid_counts(boid_counts, len(CLASSES))


