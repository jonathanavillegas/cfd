"""
Run 3D heat shield simulation (r-θ-z) with comprehensive visualization.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from cfd.solvers.heat_shield import solve_heat_shield_3d
from cfd.visualization.plot_3d import (
    plot_3d_surface_animated,
    plot_3d_isosurface,
    plot_cross_sections,
    plot_heatmap_grid,
    plot_interactive_3d
)

# Define 3D geometry
Nr = 50
Ntheta = 50
Nz = 50
R = 80
thickness = 0.05

# Time parameters
time = 100
dt = 0.1

# Material properties
k = 0.04
rho = 270
cp = 1100

# Solve
r, theta, z, transient = solve_heat_shield_3d(Nr, Ntheta, Nz, R, thickness, 
                                               time, dt, k, rho, cp)

# Convert to numpy array
transient = np.array(transient)

# Extract top surface (z = 0 layer) for animated surface plot
surface_frames = transient[:, :, :, 0]

print("Generating visualizations...")

# 1. Animated 3D surface plot of top surface
print("1. Animated 3D surface plot...")
plot_3d_surface_animated(surface_frames, r, theta, dt, title='Temperature Evolution - Top Surface')

# 2. Isosurfaces at final time
print("2. 3D isosurfaces...")
T_final = transient[-1]
T_min, T_max = np.min(T_final), np.max(T_final)
levels = np.linspace(T_min, T_max, 5)
plot_3d_isosurface(T_final, r, theta, z, levels, title='Temperature Isosurfaces (Final Time)')

# 3. Cross-sections at key times
print("3. Cross-sections...")
time_indices = [0, len(transient)//2, -1]
plot_cross_sections(transient, r, theta, z, time_indices, dt)

# 4. Depth analysis - heatmap grid
print("4. Heatmap grid at different depths...")
z_levels = [0, Nz//4, Nz//2, 3*Nz//4, Nz-1]
plot_heatmap_grid(transient[-1], r, theta, z, z_levels, time_index=-1,
                 title='Temperature at Different Depths (Final Time)')

# 5. Interactive 3D exploration
print("5. Interactive 3D visualization...")
plot_interactive_3d(transient, r, theta, z, time_index=-1)

print("All visualizations complete!")

