import pygame
import random
import math
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TKAgg')

# Constants
WIDTH, HEIGHT = 800, 800
NUM_BOIDS = 50
MAX_SPEED = 3
BOID_SIZE = 5
BOID_LENGTH = 10
BOID_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (220, 220, 220)
NEIGHBOR_RADIUS = 50
SEPARATION_RADIUS = 20
ALIGNMENT_WEIGHT = 0.05
COHESION_WEIGHT = 0.005
SEPARATION_WEIGHT = 0.05
TRUN_FACTOR = 0.3
FIGS_FOLDER = "figs"
MARGIN = {'left': 150, 'right': WIDTH-150, 'top':150, 'bottom': HEIGHT-150}
BOUNCE_OF_EDGES = False

class Boid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)

    def update(self, flock, cohesion, alignment, separation):
        dx, dy = self.compute_movement(flock, cohesion, alignment, separation)
        self.vx += dx
        self.vy += dy
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed <= 0:
            speed = 0.001
        scale = MAX_SPEED / speed
        self.vx *= scale
        self.vy *= scale
        
        self.x += self.vx
        self.y += self.vy
        self.wrap()

    def compute_movement(self, flock, cohesion, alignment, separation):
        avg_pos = [0, 0]
        avg_vel = [0, 0]
        sep = [0, 0]
        neighboring_boids = 0

        for boid in flock:
            if boid != self:
                distance = math.dist((self.x, self.y), (boid.x, boid.y))
                if distance < NEIGHBOR_RADIUS:
                    avg_pos[0] += boid.x
                    avg_pos[1] += boid.y
                    avg_vel[0] += boid.vx
                    avg_vel[1] += boid.vy
                    if distance < SEPARATION_RADIUS:
                        sep[0] += self.x - boid.x
                        sep[1] += self.y - boid.y
                    neighboring_boids += 1

        if neighboring_boids > 0:
            avg_pos[0] /= neighboring_boids
            avg_pos[1] /= neighboring_boids
            avg_vel[0] /= neighboring_boids
            avg_vel[1] /= neighboring_boids

        avg_pos[0] -= self.x
        avg_pos[1] -= self.y
        avg_vel[0] -= self.vx
        avg_vel[1] -= self.vy

        dx = alignment * avg_vel[0] + cohesion * avg_pos[0] + separation * sep[0]
        dy = alignment * avg_vel[1] + cohesion * avg_pos[1] + separation * sep[1]

        return dx, dy

    def wrap(self):
        if BOUNCE_OF_EDGES:
            if self.x < MARGIN['left']:
                self.vx = self.vx + TRUN_FACTOR
            if self.x > MARGIN['right']:
                self.vx = self.vx - TRUN_FACTOR
            if self.y > MARGIN['bottom']:
                self.vy = self.vy - TRUN_FACTOR
            if self.y < MARGIN['top']:
                self.vy = self.vy + TRUN_FACTOR
        else:
            if self.x < 0:
                self.x = WIDTH
            elif self.x > WIDTH:
                self.x = 0
            if self.y < 0:
                self.y = HEIGHT
            elif self.y > HEIGHT:
                self.y = 0
            
def draw_boids(screen, flock):
    for boid in flock:
        angle = math.atan2(boid.vy, boid.vx)
        p1 = (boid.x + BOID_LENGTH * math.cos(angle), boid.y + BOID_LENGTH * math.sin(angle))
        p2 = (boid.x + BOID_LENGTH * math.cos(angle - 2.5), boid.y + BOID_LENGTH * math.sin(angle - 2.5))
        p3 = (boid.x + BOID_LENGTH * math.cos(angle + 2.5), boid.y + BOID_LENGTH * math.sin(angle + 2.5))
        pygame.draw.polygon(screen, BOID_COLOR, (p1, p2, p3))


def pygame_sim():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Simulation")
    clock = pygame.time.Clock()

    flock = [Boid(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(NUM_BOIDS)]

    order_parameters = []
    nearest_neighbor_distances = []

    running = True
    while running:
        screen.fill(BACKGROUND_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for boid in flock:
            boid.update(flock, COHESION_WEIGHT, ALIGNMENT_WEIGHT, SEPARATION_WEIGHT)

        draw_boids(screen, flock)

        # Calculate order parameter
        avg_velocity = np.array([np.array([boid.vx, boid.vy]) / math.sqrt(boid.vx**2 + boid.vy**2) if math.sqrt(boid.vx**2 + boid.vy**2) > 0 else [0, 0] for boid in flock])
        
        order_parameter = np.linalg.norm(np.sum(avg_velocity, axis=0)) / NUM_BOIDS
        order_parameters.append(order_parameter)

        # Calculate nearest neighbor distances
        distances = []
        for boid in flock:
            distances.extend([math.dist((boid.x, boid.y), (other_boid.x, other_boid.y)) for other_boid in flock if boid != other_boid])
        nearest_neighbor_distances.append(np.mean(distances))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    return order_parameters, nearest_neighbor_distances


def main(visualise=True):
    
    if visualise: 
        order_parameters, nearest_neighbor_distances = pygame_sim()

if __name__ == "__main__":
    # Turning on visualise runs the simulation in pygame, turing it off runs the ABC algorithm
    main(visualise=True)