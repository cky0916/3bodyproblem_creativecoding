"""
Three Body Problem Visualizer
by Cathy Chen
========================================
Physics simulation of the 3 body problem, uses Verlet Integration for simulation
Simulation has 3 different eras: Stable, Figure8, and Chaotic
 - Stable: Simulate Sun, Earth, and Jupiter
 - Figure8: Special starting conditions for figure 8 pattern
 - Chaotic: Random starting positions and velocities

Requirements:
 - Python
 - pygame for rendering
 - numpy for math

Controls:
 - [E] : Cycle Era between Stable, Figure8, and Chaotic
 - [R] : Reset current Era
 - [N] : New random start
 - [T] : Toggle trails on/off

Usage:
 - Run via command line with python 3bodyproblem.py
 - Run in VSCode by hitting run button
"""
#--------------------IMPORTS-------------------------------
import sys #used to exit the program
import math #math, random number generation
import time # to generate seeds
import random #random number generation/seed setting
from dataclasses import dataclass #to code datastructure Bodies
import numpy as np #math library
import pygame #visualization library

#-----------------------CONFIG--------------------------------
"""
This section contains global constants to tune the simulation
scale and the physics accuracy. The most important one here is 
the SOFTENING, which is added to distances to avoid divide by zero
"""

WIDTH, HEIGHT = 1280, 720 #Window Size: Width 1280 x Height 720
BG_COLOR = (5, 7, 12) # Background Color: near-black blue
G_CONST = 1.0 # Scaled gravitational constant: gravitational constant (scaled units)
SOFTENING = 1e-2 # Softening: no div by zero errors
MAX_TRAIL_ALPHA = 1 # ALPHA: Per frame fade on trails
COLORS = [(90, 200, 255), (255, 145, 80), (190, 140, 255)] #color palette of the planets
stable = False # I use this variable to check if I'm on stable era to turn down the dt to slow simulation down a bit
# ---------------------------- Physics Core ---------------------------

@dataclass
class Bodies:
    # object to keep track of planets and their numbers
    # contains the positions of all the planets, velocities, and masses
    pos: np.ndarray # position array
    vel: np.ndarray # velocity array
    mass: np.ndarray # mass array

    def copy(self) -> "Bodies":
        return Bodies(self.pos.copy(), self.vel.copy(), self.mass.copy())

def accelerations(pos: np.ndarray, mass: np.ndarray, eps2: float = SOFTENING) -> np.ndarray:
    #the accelerations are calculated here for a single point with respect to all other bodies
    #this is since all planets pull on each other
    # calculate position of a body with respect to all other bodies
    r = pos[np.newaxis, :, :] - pos[:, np.newaxis, :] 
    # compute squared distances of each pair, add a very small softening on diagonal so calculation doesn't explode
    dist2 = np.sum(r * r, axis=2) + np.eye(pos.shape[0]) * 1e-12 
    # add in the softening to avoid massive forces when close
    dist2 += eps2
    #compute 1/r^3 needed for gravitational force
    inv_dist3 = 1.0 / (np.power(dist2, 1.5))
    #reshape the mass array to apply over all pairs
    mcol = mass[np.newaxis, :, np.newaxis]
    #calculate acceleration:
    #acceleration on body i = G * sum of all accelerations pulling from each other body j
    acc = G_CONST * np.sum(r * inv_dist3[:, :, np.newaxis] * mcol, axis=1)
    return acc

def verlet_step(b: Bodies, dt: float):
    #one step of verlet velocity integration
    #https://en.wikipedia.org/wiki/Verlet_integration
    #compute accelerations
    a0 = accelerations(b.pos, b.mass)
    #advance velocities by half a timestep
    b.vel += 0.5 * dt * a0
    #update positions
    b.pos += dt * b.vel
    #compute accelerations again at new positions
    a1 = accelerations(b.pos, b.mass)
    #update velocities by the remaining half timestep
    b.vel += 0.5 * dt * a1


# ---------------------------- Presets / Eras -------------------------
"""
This section contains all the preset functions for each of the eras:
- Stable: Heavy sun, lighter jupiter and Earth, circular orbits
- Figure8: Solution where 3 equal masses go in a figure8 path
- Chaotic: Random starting positions and velocities
"""
def era_stable() -> Bodies:
    # this preset creates a simple "solar system like" configuration:
    # very heavy central object (sun) and 2 smaller planets (earth and jupiter)

    #stable era, change stable to True
    global stable
    stable = True

    m_sun, m_jup, m_earth = 350.0, 0.5, 0.5  # #masses chosen for sun, jupiter, earth (not accurate masses)


    #setting their initial positions
    pos = np.array([
        [0.0, 0.0],   # sun at origin
        [4.2, 0.0],   # jupiter 3.2 away
        [1.0, 0.0],   # earth 1 away
    ], dtype=np.float64)

    #setting the masses array
    mass = np.array([m_sun, m_jup, m_earth], dtype=np.float64)

    def circ_vel(M, r):
        #calculate circular orbit speed for planets
        #this comes from setting gravitational force = centripetal force
        # v = sqrt(G*M / r)
        return math.sqrt(G_CONST * M / (r + 1e-9))

    #give jupiter and earth orbit velocity so they stay in orbit
    #sun will have velocity of 0
    v_j = circ_vel(m_sun, 5.2)
    v_e = circ_vel(m_sun, 1.0)

    vel = np.array([
        [0.0, 0.0],          # sun has initially 0 velocity
        [0.0, v_j],   # jupiter has a little reduced speed so it looks more stable
        [0.0, v_e],  # earth gets a bit boosted speed so it looks more dynamic
    ], dtype=np.float64)

    # shifting to center of mass frame
    # this removes overall drift
    total_m = mass.sum()
    v_cm = (vel * mass[:, None]).sum(axis=0) / total_m
    vel -= v_cm

    return Bodies(pos, vel, mass)


def era_figure8() -> Bodies:
    #figure 8 preset, all bodies follow the same figure 8 path
    #starting conditions done through ChatGPT

    #not a stable era, change stable back to False
    global stable
    stable = False

    #all bodies have equal mass here
    mass = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    #ChatGPT initial positions
    pos = np.array([
        [-0.97000436,  0.24308753],
        [ 0.97000436, -0.24308753],
        [ 0.0,         0.0       ],
    ], dtype=np.float64)

    #ChatGPT initial velocities
    vel = np.array([
        [ 0.4662036850,  0.4323657300],
        [ 0.4662036850,  0.4323657300],
        [-0.93240737,   -0.86473146  ],
    ], dtype=np.float64)

    #convert into a center of mass frame
    total_m = mass.sum()
    v_cm = (vel * mass[:, None]).sum(axis=0) / total_m
    vel -= v_cm
    r_cm = (pos * mass[:, None]).sum(axis=0) / total_m
    pos -= r_cm

    return Bodies(pos, vel, mass)


def era_chaotic(seed: int | None = None) -> Bodies:

    #not a stable era, change stable back to False
    global stable
    stable = False

    #setting seed if given
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    #equal masses in this case
    mass = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    #random positions by doing random angles and radius
    r_scale = 1.5
    angles = np.random.uniform(0, 2 * math.pi, size=3)
    radii = np.random.uniform(0.6 * r_scale, 1.2 * r_scale, size=3)

    #convert polar to x, y
    pos = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1).astype(np.float64)

    # Velocities done by ChatGPT
    # for each body, we're trying to give a velocity roughly perpendicular to radius
    # this is so that we get roughly orbital
    vel = np.zeros_like(pos)

    for i in range(3):
        #get the position
        r = pos[i]

        # perpendicular direction
        perp = np.array([-r[1], r[0]])

        #normalize to a unit vector for the velocity
        norm = np.linalg.norm(perp) + 1e-9
        perp /= norm

        #multiply to a random speed
        speed = np.random.uniform(0.2, 0.9)

        #add random jitter
        jitter = np.random.uniform(-0.15, 0.15, size=2)

        vel[i] = speed * perp + jitter

    #remove center of mass drift so system stays in place
    total_m = mass.sum()
    v_cm = (vel * mass[:, None]).sum(axis=0) / total_m
    vel -= v_cm

    r_cm = (pos * mass[:, None]).sum(axis=0) / total_m
    pos -= r_cm

    return Bodies(pos, vel, mass)


#dictionary of names to presets
ERAS = [
    ("Stable Era", era_stable),
    ("Figure-8 Era", era_figure8),
    ("Chaotic Era", era_chaotic),
]

# ---------------------------- SIMULATION ------------------------------
"""
This section contains the setup of the simulation window (pygame) and the 
simulation loop itself, which handles the window itself, the inputs, and the 
main drawing loop for bodies and trails.
"""
class threebody:
    def __init__(self):
        #initialize the pygame
        pygame.init()
        #create the main window where everything is drawn
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        #caption the window
        pygame.display.set_caption("Three Body Problem")
        #internal clock to take care of frames per second
        self.clock = pygame.time.Clock()

        #scale to convert world units into pixels on the screen
        self.scale = 120.0
        #starting place of camera is center of screen
        self.offset = np.array([WIDTH/2, HEIGHT/2])

        # separate surfaces
        self.trail_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA) #store trails on this surface
        self.fade_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA) #fade the trails on this one
        self.fade_surf.fill((0, 0, 0, MAX_TRAIL_ALPHA)) #this works by filling with semi transparent black that we overlay over and over
        self.show_trails = True

        #simulation start
        self.era_idx = 0
        self.reset_era(new_era=False)

    def reset_era(self, new_era=False, force_random=False):
        #reset the simulation to a chosen state
        if new_era:
            self.era_idx = (self.era_idx + 1) % len(ERAS) #next era on eras list
        
        #get the eras name and its function/preset
        name, factory = ERAS[self.era_idx]
        
        #random starting conditions
        if force_random:
            seed = int(time.time() * 1000) & 0xffffffff
            self.bodies = era_chaotic(seed=seed)
            print(f"Mode: Randomized Chaotic (Seed: {seed})")
        else:
            # use era preset, or random starting if "Chaotic Era"
            seed = (int(time.time() * 1000) & 0xffffffff) if name == "Chaotic Era" else None
            self.bodies = factory(seed=seed) if seed else factory()
            print(f"Mode: {name}")

        self.trail_surf.fill((0,0,0,0))

    def run(self):
        #main loop of the app
        while True:
            # user inputs
            for event in pygame.event.get():
                if event.type == pygame.QUIT: #if we exit out, shut down pygame and end program
                    sys.exit()
                elif event.type == pygame.KEYDOWN: #if we press a key, check what key was pressed
                    if event.key == pygame.K_e: #keypress E: cycle era
                        self.reset_era(new_era=True)
                    elif event.key == pygame.K_r: #keypress R: reset era
                        self.reset_era(new_era=False)
                    elif event.key == pygame.K_n: #keypress N: random starting condition
                        self.reset_era(force_random=True)
                    elif event.key == pygame.K_t: #keypress T: start trails
                        self.show_trails = not self.show_trails
                        if not self.show_trails:
                            self.trail_surf.fill((0,0,0,0))

            # update positions with verlet step function
            for _ in range(4): # 4 steps per frame

                #if we're looking at stable, turn down dt to slow down sim
                global stable
                if stable:
                    verlet_step(self.bodies, dt=0.001)
                #otherwise, keep dt as 0.01
                else:
                    verlet_step(self.bodies, dt=0.01)
                

            # fill in the background to erase canvas
            self.screen.fill(BG_COLOR)

            # draw in the trails
            if self.show_trails:
                #slightly darken the older trails (fade out)
                #add in the new trails
                self.trail_surf.blit(self.fade_surf, (0, 0))
                # overlay the updated trails onto main screen
                self.screen.blit(self.trail_surf, (0, 0))

            # project it onto the screen
            screen_pos = (self.bodies.pos * self.scale + self.offset).astype(int)

            # draw in the bodies
            for i, p in enumerate(screen_pos):
                #pick a color from the colors list
                color = COLORS[i % len(COLORS)]
                #draw a circle onto the screen with the colors list
                pygame.draw.circle(self.screen, color, p, 4)
                
                #draw a smaller dot on the trail surface to leave a trail
                if self.show_trails:
                    pygame.draw.circle(self.trail_surf, color, p, 2)

            #push the frame to the screen so it becomes visible
            pygame.display.flip()

            #limit frame rate to 60 fps
            self.clock.tick(60)

if __name__ == "__main__":
    threebody().run()