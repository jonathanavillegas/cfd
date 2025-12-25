import numpy as np
import matplotlib.pyplot as plt
import helper as f
import matplotlib.animation as animation



#time in seconds
time = 100
dt = .05

#define 2D geometry
numXNodes = 50
numYNodes = 50
num_nodes = numXNodes * numYNodes
xL = 10 
yL = 10
x = np.linspace(0,xL, numXNodes)
y = np.linspace(0,yL, numYNodes)
dx = xL / (numXNodes - 1)
dy = yL / (numYNodes - 1)
k = 700
rho = 1000
cp = 10
T0 = 0

#heat flux array
q = np.zeros(num_nodes)
q[numXNodes * 25 + 25] = 10000000

#store BCs
BCW_type = "dirichlet"
SW = f.def_BCSolution(BCW_type, 10, dx, k, 0)

BCE_type = "dirichlet"
SE = f.def_BCSolution(BCE_type, 10, dx, k, 0)

BCN_type = "dirichlet"
SN = f.def_BCSolution(BCN_type, 10, dy, k, 0)

BCS_type = "dirichlet"
SS = f.def_BCSolution(BCS_type, 10, dy, k, 0)

#build initial condition; start with constant temperature
initial_condition = f.initial_condition(num_nodes, numXNodes, numYNodes, T0, SW, SE, SN, SS)
print(len(initial_condition))
#solve for next time step and store; account for BC; we will start with dirichlet BCs for simplicity
answer = np.array(f.transient_run(time, dt, num_nodes, numXNodes, numYNodes, initial_condition, k, rho, cp, dx, dy, SW, SE, SN, SS, q))
answer_3D = np.array([sol.reshape((numYNodes, numXNodes)) for sol in answer])

#iterativvely do this until end of duration or settles out into SSS
#animation

fig, ax = plt.subplots()
cax = ax.imshow(answer_3D[0], cmap='hot', origin='lower', extent=[0, xL, 0, yL])
fig.colorbar(cax, label='Temperature')
ax.set_title("Time = 0.0 s")
ax.set_xlabel("X")
ax.set_ylabel("Y")

def update(frame):
    cax.set_array(answer_3D[frame])
    ax.set_title(f"Time = {frame * dt:.2f} s")
    return [cax]

ani = animation.FuncAnimation(
    fig, update, frames=answer_3D.shape[0], interval=10, blit=False
)

plt.show()
