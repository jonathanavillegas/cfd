import numpy as np
import matplotlib.pyplot as plt
import helper as f
import matplotlib.animation as animation

#define 2D geometry
Nr = 50
Ntheta = 50
Nz = 50

num_nodes = Nr * Ntheta * Nz
R = 80 
thickness = .05
dr = R / (Nr - 1)
dtheta = 2*np.pi / (Ntheta - 1)
dz = thickness / Nz

# Create flux distribution (your function)
r, theta, q = f.create_flux_distribution(Nr, Ntheta)  # q shape = (Nr, Ntheta)

r_edges = np.linspace(0, R, Nr + 1)
theta_edges = np.linspace(0, 2 * np.pi, Ntheta + 1)

R_grid, Theta_grid = np.meshgrid(r_edges, theta_edges, indexing='ij')

X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# Plot using pcolormesh with correct shape
'''plt.figure(figsize=(6, 6))
plt.pcolormesh(X, Y, q, shading='auto', cmap='hot')  # 'auto' avoids shape mismatch
plt.colorbar(label='Heat flux (kW/m²)')
plt.title('Heat Flux Distribution on Circular Disk')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.axis('equal')
plt.show()
'''

time = 100
dt = .1
k = 0.04        
rho = 270        
cp = 1100        


initial_condition = np.zeros((Nr, Ntheta, Nz))
answer = np.array(f.transient_run_3D(time, Nr, Ntheta, Nz, dr, dtheta, dz, dt, initial_condition, q, r, k, rho, cp))
#Stopping point; it runs but need to figure out plotting
# Extract top surface (z = 0 layer)
z_index = 0
surface_frames = np.array([frame[:, :, z_index] for frame in answer])  # shape: (Nt, Nr, Ntheta

# Fix θ index to 0
theta_index = 0

# Slice shape: (Nt, Nr, Nz)
rz_slice = np.array([frame[:, theta_index, :] for frame in answer])

r_vals = np.linspace(0, R, Nr)
z_vals = np.linspace(0, thickness, Nz)
R_grid, Z_grid = np.meshgrid(r_vals, z_vals, indexing='ij')  # shape (Nr, Nz)

fig, ax = plt.subplots()
c = ax.imshow(rz_slice[0], origin='lower', aspect='auto', extent=(0, thickness, 0, R), cmap='hot')
fig.colorbar(c, ax=ax)
ax.set_title('r-z Cross Section of Temperature')
ax.set_xlabel('z [m]')
ax.set_ylabel('r [mm]')

def update(frame):
    c.set_array(rz_slice[frame])
    ax.set_title(f'Time = {frame * dt:.2f} s')
    return [c]

ani = animation.FuncAnimation(fig, update, frames=len(rz_slice), interval=100)
plt.show()




