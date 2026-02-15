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
Nr = 50  # [-] Number of radial nodes
Ntheta = 50  # [-] Number of angular nodes
distribution_type = 1  # [-] Heat flux distribution: 1 = Original (R=75mm), 2 = Vehicle shape (R=0.4m)
# Radius automatically set based on distribution type
if distribution_type == 1:
    R = 0.075  # [m] Outer radius for distribution 1 (75 mm)
elif distribution_type == 2:
    R = 0.4  # [m] Outer radius for distribution 2 (400 mm vehicle scale)
else:
    R = 0.075  # [m] Default to distribution 1 radius (75 mm)

# Time parameters
time = 100  # [s] Total simulation time
dt = 0.1  # [s] Time step size

# Material properties
k = 0.04  # [W/(m·K)] Thermal conductivity
rho = 270  # [kg/m³] Density
cp = 1100  # [J/(kg·K)] Specific heat capacity

# Solve
# Returns: r [m], theta [rad], transient [K] - Temperature field at each time step
r, theta, transient = solve_heat_shield_2d(Nr, Ntheta, R, time, dt, k, rho, cp, 
                                            distribution_type=distribution_type)

# Visualize
plot_cylindrical_2d_animated(transient, r, theta, dt, title='Temperature Evolution')

