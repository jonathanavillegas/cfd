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
from cfd.visualization.plot_3d import plot_rz_cross_sections, plot_surface_temperature_evolution

# Define 3D geometry
Nr = 50  # [-] Number of radial nodes
Ntheta = 50  # [-] Number of angular nodes
Nz = 50  # [-] Number of axial nodes
distribution_type = 1  # [-] Heat flux distribution: 1 = Original (R=75m), 2 = Vehicle shape (R=0.4m)
# Radius automatically set based on distribution type
if distribution_type == 1:
    R = 75.0  # [m] Outer radius for distribution 1
elif distribution_type == 2:
    R = 0.4  # [m] Outer radius for distribution 2 (400 mm vehicle scale)
else:
    R = 75.0  # [m] Default to distribution 1 radius
thickness = 0.05  # [m] Shield thickness

# Time parameters
time = 10  # [s] Total simulation time
dt = .1  # [s] Time step size

# Material properties
k = 0.04  # [W/(m·K)] Thermal conductivity
rho = 270  # [kg/m³] Density
cp = 1100  # [J/(kg·K)] Specific heat capacity

# Solve
# Returns: r [m], theta [rad], z [m], transient [K] - Temperature field at each time step
r, theta, z, transient = solve_heat_shield_3d(Nr, Ntheta, Nz, R, thickness, 
                                               time, dt, k, rho, cp, 
                                               distribution_type=distribution_type)

# Convert to numpy array
transient = np.array(transient)

print("Generating surface temperature evolution visualization...")

# Plot surface temperature evolution (r-θ plane at z=0)
plot_surface_temperature_evolution(transient, r, theta, dt, z_surface_idx=0, 
                                    title='Surface Temperature Evolution')

print("Visualization complete!")

