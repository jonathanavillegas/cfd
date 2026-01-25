"""
Run 3D heat shield simulation (r-θ-z) with r-z cross-section visualization.

Shows temperature evolution in the radial-depth plane at multiple time steps.
Time intervals are automatically adjusted based on total simulation time.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from cfd.solvers.heat_shield import solve_heat_shield_3d
from cfd.visualization.plot_3d import plot_rz_cross_sections

# Define 3D geometry
Nr = 50
Ntheta = 50
Nz = 50
R = 80
thickness = 0.05

# Time parameters
time = 100
dt = 1

# Material properties
k = 0.04
rho = 270
cp = 1100

# Solve
r, theta, z, transient = solve_heat_shield_3d(Nr, Ntheta, Nz, R, thickness, 
                                               time, dt, k, rho, cp)

# Convert to numpy array
transient = np.array(transient)

print("Generating r-z cross-section visualization...")

# Plot r-z cross-sections at multiple time steps with adaptive intervals
plot_rz_cross_sections(transient, r, theta, z, time, dt, theta_idx=0)

print("Visualization complete!")

