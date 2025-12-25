"""
Run 2D transient heat conduction simulation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from cfd.solvers.transient_2d import solve_2d_transient
from cfd.visualization.plot_2d import plot_2d_animated

# Time in seconds
time = 100
dt = 0.05

# Define 2D geometry
numXNodes = 50
numYNodes = 50
xL = 10
yL = 10
k = 700
rho = 1000
cp = 10
T0 = 0

# Solve
x, y, transient = solve_2d_transient(numXNodes, numYNodes, xL, yL, time, dt, 
                                    k, rho, cp, T0)

# Reshape for visualization
answer_3D = np.array([sol.reshape((numYNodes, numXNodes)) for sol in transient])

# Animate
plot_2d_animated(answer_3D, x, y, dt, title='Temperature Evolution')

