# Three-Body Problem: Visualizing Chaos

An interactive visualization of the Three-Body Problem, inspired by Cixin Liu's book of the same name and Chaos Theory, such as Lorenz Attractors.

---

## Overview

- Simulates three planets with masses interacting under the influence of gravity.
- Includes trails to visualize the paths that the planets take
- All code is contained in a single file: `3bodyproblem.py`.

---

## Requirements

- Python 3.8+
- Packages:
  - `pygame`
  - `numpy`

---

## Quick Start

From the folder with `3bodyproblem.py`:

```
python 3bodyproblem.py
```

This opens a pygame window and starts the simulation.

---

## Controls

Keyboard:

* [E] – Cycle to the next era (Stable, Figure-8, Chaotic, ...)
* [R] – Reset the current era to its default initial conditions
* [N] – Create a new random starting condition (same as Chaotic)
* [T] – Turn trails on/off

---

## Eras (Presets)

The simulation includes 3 built-in presets:

1. **Stable Era**

   * One Sun with high mass, and 2 lighter planets, Jupiter and Earth
   * Initial positions and velocities are set so that they stay in orbit

2. **Figure-8 Era**

   * Three identical planets
   * Hard-coded start conditions to make the 3 planets go in a figure 8 pattern

3. **Chaotic Era**

   * Three identical planets
   * Random ish start conditions
   * This is the **Chaos** thats being visualized
  
Use **E**, **R**, and **N** to go between these setups

---

## Code Structure:

The code is written into a few sections:
_Note that this is just a high level overview, more detailed documentation is in the code_
1. **Imports & Config**

   * Imports `sys`, `math`, `time`, `random`, `dataclasses`, `numpy`, and `pygame`
   * Sets variables for window size, background color, gravitational constant, softening, trail fading, and body colors

2. **Physics Core**

   * `Bodies` keeps track of positions, velocities, and masses for all planets
   * `accelerations(pos, mass)` calculates the acceleration on each planet.
   * `verlet_step(bodies, dt)` moves the system forward by one time step, updating the position of the planets

3. **Preset / Era Functions**

   * `era_stable()` builds the "Stable" era, solar system like
   * `era_figure8()` builds the figure8 configuration
   * `era_chaotic(seed=None)` builds a random chaotic configuration

4. **Simulation Class (`threebody`)**

   * Handles all pygame initialization and setup
   * Controls the camera and the offset, as well as 2 surfaces

     * a main screen surface and a separate trail surface (to fade the trails)
   * `reset_era(new_era=False, force_random=False)` chooses a preset, clears trails, and sets up new simulation
   * `run()` is the main loop:
     * Watches for keyboard actions
     * Moves the simulation forward using multiple `verlet_step` calls
     * Draws the background, trails, and planets
     * Shows to user, caps frame rate at 60fps


