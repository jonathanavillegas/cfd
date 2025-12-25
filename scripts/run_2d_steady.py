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
k = 0.001

# Solve
x, y, T = solve_2d_steady(numXNodes, numYNodes, xL, yL, k)

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(T, extent=[0, xL, 0, yL], origin='lower', cmap='viridis')
plt.colorbar(label='Temperature')
plt.title('Steady State Heat Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(False)
plt.show()

