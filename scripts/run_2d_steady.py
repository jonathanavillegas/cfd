"""
Run 2D steady-state heat conduction simulation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from cfd.solvers.poisson_2d import solve_2d_steady

# Define 2D geometry
numXNodes = 75  # [-] Number of nodes in x-direction
numYNodes = 75  # [-] Number of nodes in y-direction
xL = 10  # [m] Domain length in x-direction
yL = 10  # [m] Domain length in y-direction
k = .01  # [W/(m·K)] Thermal conductivity

# Define boundary conditions
BCW_type = "neumann"  # [-] West boundary type: "dirichlet" or "neumann"
BCE_type = "neumann"  # [-] East boundary type: "dirichlet" or "neumann"
BCN_type = "dirichlet"  # [-] North boundary type: "dirichlet" or "neumann"
BCS_type = "dirichlet"  # [-] South boundary type: "dirichlet" or "neumann"
source_x = 5.0  # [m] Source x-coordinate location
source_y = 5.0  # [m] Source y-coordinate location
source_strength = 100  # [W/m³] Volumetric heat generation rate

# Define boundary values; Temperature for Dirichlet, Flux for Neumann
SW = 5  # [K] for Dirichlet, [W/m²] for Neumann - West boundary value
SE = -5  # [K] for Dirichlet, [W/m²] for Neumann - East boundary value
SN = 100  # [K] for Dirichlet, [W/m²] for Neumann - North boundary value
SS = 100  # [K] for Dirichlet, [W/m²] for Neumann - South boundary value

# Solve
# Returns: x [m], y [m], T [K] - Temperature field
x, y, T = solve_2d_steady(numXNodes, numYNodes, xL, yL, k, BCW_type, BCE_type, BCN_type, BCS_type, source_x, source_y, source_strength, SW, SE, SN, SS)

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(T, extent=(0, xL, 0, yL), origin='lower', cmap='viridis')
plt.colorbar(label='Temperature [K]')
plt.title('Steady State Heat Distribution')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.grid(False)
plt.show()

