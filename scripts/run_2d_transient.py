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

# Define boundary conditions
BCW_type = "nuemann"
BCE_type = "nuemann"
BCN_type = "dirichlet"
BCS_type = "nuemann"
SW = 5
SE = -10
SN = 0 
SS = -5
source_x = 5
source_y = 5
source_strength = 500

# Solve
x, y, transient = solve_2d_transient(numXNodes, numYNodes, xL, yL, time, dt, k, rho, cp, T0,
                                     BCW_type, BCE_type, BCN_type, BCS_type, SW, SE, SN, SS,
                                     source_x, source_y, source_strength)

# Reshape for visualization
answer_3D = np.array([np.array(sol).reshape((numYNodes, numXNodes)) for sol in transient])

# Animate
plot_2d_animated(answer_3D, x, y, dt, title='Temperature Evolution')

