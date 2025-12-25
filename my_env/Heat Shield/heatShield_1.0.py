import numpy as np
import matplotlib.pyplot as plt
import helper as f
import matplotlib.animation as animation

#define 2D geometry
Nr = 50
Ntheta = 50
Nt = 50

num_nodes = Nr * Ntheta
R = 80 
thickness = .5
dr = R / (Nr - 1)
dtheta = 2*np.pi / (Ntheta - 1)
dt = thickness / Nt
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
dt = .01
k = 0.04        
rho = 270        
cp = 1100        


initial_condition = np.zeros((Nr, Ntheta))
answer = np.array(f.transient_run_2D(time, Nr, Ntheta, dr, dtheta, dt, initial_condition, q, r, k, rho, cp))

fig, ax = plt.subplots(figsize=(6, 6))
c = ax.pcolormesh(X, Y, answer[0], shading='auto', cmap='hot', vmin=np.min(answer), vmax=np.max(answer))
fig.colorbar(c, ax=ax)
ax.set_title('Temperature Evolution')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_aspect('equal')

def animate(frame):
    c.set_array(answer[frame].flatten())
    ax.set_title(f'Time: {frame * dt}')
    return c,

ani = animation.FuncAnimation(fig, animate, frames=len(answer), interval=dt)
plt.show()
                            



