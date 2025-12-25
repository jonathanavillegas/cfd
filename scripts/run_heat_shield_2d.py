"""
Run 2D heat shield simulation (r-θ plane).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from cfd.solvers.heat_shield import solve_heat_shield_2d
from cfd.visualization.plot_2d import plot_cylindrical_2d_animated

# Define 2D geometry
Nr = 50
Ntheta = 50
R = 80

# Time parameters
time = 100
dt = 0.01

# Material properties
k = 0.04
rho = 270
cp = 1100

# Solve
r, theta, transient = solve_heat_shield_2d(Nr, Ntheta, R, time, dt, k, rho, cp)

# Visualize
plot_cylindrical_2d_animated(transient, r, theta, dt, title='Temperature Evolution')

