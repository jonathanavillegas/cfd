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
numXNodes = 50
numYNodes = 50
xL = 10
yL = 10
k = .01

# Define boundary conditions
BCW_type = "neumann"
BCE_type = "neumann"
BCN_type = "dirichlet"
BCS_type = "dirichlet"
source_i = 5
source_j = 5
source_strength = 10

# Define boundary values; Temperature for Dirichlet, Flux for Neumann
SW = 5
SE = -5
SN = 100
SS = 100

# Solve
x, y, T = solve_2d_steady(numXNodes, numYNodes, xL, yL, k, BCW_type, BCE_type, BCN_type, BCS_type, source_i, source_j, source_strength, SW, SE, SN, SS)

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(T, extent=(0, xL, 0, yL), origin='lower', cmap='viridis')
plt.colorbar(label='Temperature')
plt.title('Steady State Heat Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(False)
plt.show()

