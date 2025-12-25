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
num_nodes = 100
length = 10
k = 5
BCR_type = "neumann"

# Solve
x, T = solve_1d_steady(num_nodes, length, k, BCR_type=BCR_type)

# Plot
plt.plot(x, T)
plt.xlabel('Position')
plt.ylabel('Temperature')
plt.title('1D Steady-State Heat Distribution')
plt.grid(True)
plt.show()

