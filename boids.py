import numba as nb
import numpy as np
import pygame
import math
np.random.seed(0)

WIDTH, HEIGHT = 800, 800

CLASSES = np.array([200, 10])
NUM_BOIDS = np.sum(CLASSES)
VISIBLE_RADIUS = np.array([[50, 50],
                          [50, 50]])
SEPARATION_RADIUS = np.array([[10, 10],
                             [10, 10]])
ALIGNMENT_WEIGHT = np.array([[0.05, 0.05], 
                             [0.05, 0.05]])
COHESION_WEIGHT = np.array([[0.005, 0], 
                            [0.005, 0.005]])
SEPARATION_WEIGHT = np.array([[0.1, 0.9], 
                              [0.1, 0.1]])

TRUN_FACTOR = 0.2
BOID_LENGTH = [10, 14]
BACKGROUND_COLOR = (220, 220, 220)
BOID_COLOR = [(0, 0, 0), (255, 0, 0)]

MAX_SPEED = 3
MARGIN_LEFT=100; MARGIN_RIGHT=WIDTH-100; MARGIN_TOP=100; MARGIN_BOTTOM=HEIGHT-100

@nb.njit
def frobenius_norm(a):
    norms = np.empty(a.shape[0], dtype=a.dtype)
    for i in nb.prange(a.shape[0]):
        norms[i] = np.sqrt(a[i, 0] * a[i, 0] + a[i, 1] * a[i, 1])
    return norms

@nb.njit
def update_numba(boids, classes):
    for i in range(len(boids)):
        # Calculate the angle between the current boid and the neighbor
        angle_to_neighbor = np.arctan2(boids[:, 1] - boids[i, 1], boids[:, 0] - boids[i, 0])
        # Calculate the angle difference between the current boid's velocity direction and the angle to the neighbor
        angle_difference = np.abs(np.arctan2(boids[i, 3], boids[i, 2]) - angle_to_neighbor)
        # 1 = 90 degrees
        # 2 = 180 degrees
        # 3 = 270 degrees
        # 4 = 360 degrees
        vision_angle = math.pi * 3
        # Check if the neighbor is within the vision angle range
        angle_mask = angle_difference <= vision_angle / 2
        distances = frobenius_norm(boids[i, :2] - boids[:, :2])

        for c in range(len(CLASSES)):
            class_mask = classes == c
            class_mask = np.stack((class_mask, class_mask), axis=1)

            # Use makes to find neighbors
            sep_mask = distances < SEPARATION_RADIUS[classes[i], c]
            sep_mask = np.logical_and(sep_mask, angle_mask)
            sep_mask = np.stack((sep_mask, sep_mask), axis=1)
            viaible_mask = np.logical_and(SEPARATION_RADIUS[classes[i], c] <= distances, distances < VISIBLE_RADIUS[classes[i], c], angle_mask)
            viaible_mask = np.stack((viaible_mask, viaible_mask), axis=1)
            
            num_neighbors = np.sum(viaible_mask*class_mask)/2

            # Separation
            boids[i, 2:] = boids[i, 2:] + np.sum((boids[i, :2] - boids[:, :2])*sep_mask*class_mask, axis=0) * SEPARATION_WEIGHT[classes[i], c]
            
            # Check if neighboring_boids>0
            if num_neighbors > 0:
                # Alignment
                boids[i, 2:] = boids[i, 2:] + (np.sum((boids[:, 2:])*viaible_mask*class_mask, axis=0) / num_neighbors - boids[i, 2:]) * ALIGNMENT_WEIGHT[classes[i], c]

                # Cohesion
                boids[i, 2:] = boids[i, 2:] + (np.sum((boids[:, :2])*viaible_mask*class_mask, axis=0) / num_neighbors - boids[i, :2]) * COHESION_WEIGHT[classes[i], c]

    # Turn around a screen edges
    boids[:, 2] = boids[:, 2] + (boids[:, 0] < MARGIN_LEFT) * TRUN_FACTOR
    boids[:, 2] = boids[:, 2] - (boids[:, 0] > MARGIN_RIGHT) * TRUN_FACTOR
    boids[:, 3] = boids[:, 3] - (boids[:, 1] > MARGIN_BOTTOM) * TRUN_FACTOR
    boids[:, 3] = boids[:, 3] + (boids[:, 1] < MARGIN_TOP) * TRUN_FACTOR

    # Normalize speed
    speed = np.sqrt(boids[:, 2]**2 + boids[:, 3]**2)[:, np.newaxis]
    scale = MAX_SPEED / speed
    boids[:, 2:] = boids[:, 2:] * scale

    # Update
    boids[:, :2] = boids[:, :2] + boids[:, 2:]

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
    
    classes = np.concatenate([[i] * number for i, number in enumerate(CLASSES)])

    running = True
    while running:
        screen.fill(BACKGROUND_COLOR)
        draw_dotted_margin(screen, WIDTH, HEIGHT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Running the update with namba gives some initial wait time, 
        # becuase the functions has to be converted and chached, but is able to run large numbers of boids
        update_numba(boids, classes)

        draw_boids(screen, boids, classes)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    pygame_sim()


