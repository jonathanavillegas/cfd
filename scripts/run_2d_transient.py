"""
Run 2D transient heat conduction simulation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from cfd.solvers.transient_2d import solve_2d_transient
from cfd.visualization.plot_2d import plot_2d_animated

# Time parameters
time = 100  # [s] Total simulation time
dt = 0.05  # [s] Time step size

# Define 2D geometry
numXNodes = 50  # [-] Number of nodes in x-direction
numYNodes = 50  # [-] Number of nodes in y-direction
xL = 10  # [m] Domain length in x-direction
yL = 10  # [m] Domain length in y-direction
k = 1000  # [W/(m·K)] Thermal conductivity
rho = 1000  # [kg/m³] Density
cp = 10  # [J/(kg·K)] Specific heat capacity
T0 = 0  # [K] Initial temperature

# Define boundary conditions
BCW_type = "neumann"  # [-] West boundary type: "dirichlet" or "neumann"
BCE_type = "neumann"  # [-] East boundary type: "dirichlet" or "neumann"
BCN_type = "neumann"  # [-] North boundary type: "dirichlet" or "neumann"
BCS_type = "neumann"  # [-] South boundary type: "dirichlet" or "neumann"
SW = 5  # [K] for Dirichlet, [W/m²] for Neumann - West boundary value
SE = 5  # [K] for Dirichlet, [W/m²] for Neumann - East boundary value
SN = -5  # [K] for Dirichlet, [W/m²] for Neumann - North boundary value
SS = -5  # [K] for Dirichlet, [W/m²] for Neumann - South boundary value
source_x = 5  # [m] Source x-coordinate location
source_y = 5  # [m] Source y-coordinate location
source_strength = 250  # [W/m³] Volumetric heat generation rate

# Solve
# Returns: x [m], y [m], transient [K] - Temperature field at each time step
x, y, transient = solve_2d_transient(numXNodes, numYNodes, xL, yL, time, dt, k, rho, cp, T0,
                                     BCW_type, BCE_type, BCN_type, BCS_type, SW, SE, SN, SS,
                                     source_x, source_y, source_strength)

# Reshape for visualization
answer_3D = np.array([np.array(sol).reshape((numYNodes, numXNodes)) for sol in transient])

plot_2d_animated(answer_3D, x, y, dt, title='Temperature Evolution')

