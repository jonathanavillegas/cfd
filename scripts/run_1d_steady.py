"""
Run 1D steady-state heat conduction simulation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from cfd.solvers.poisson_1d import solve_1d_steady

# Define 1D geometry
num_nodes = 1000
length = 10
k = 5
BCL_type = "dirichlet"
BCR_type = "neumann"
sl = 0
sr = 10
source_pos = 5
source_strength = 10

# Solve
x, T = solve_1d_steady(num_nodes, length, k, BCL_type, BCR_type, sl, sr, source_pos, source_strength)

# Plot
plt.plot(x, T)
plt.xlabel('Position')
plt.ylabel('Temperature')
plt.title('1D Steady-State Heat Distribution')
plt.grid(True)
plt.show()

