#solver should be able to solve for Poisson EQ (steady state heat with constant heat transfer coefficient)
import numpy as np
import matplotlib.pyplot as plt
import helper as f

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
k = .001




# Build our solution veector given BCs and desired source value
S = f.def_forceFunction(num_nodes, 25, 25, k, 1000, numXNodes)

BCW_type = "nuemann"
SW = f.def_BCSolution(BCW_type, 0, dx, k, 5)

BCE_type = "nuemann"
SE = f.def_BCSolution(BCE_type, 0, dx, k, -5)

BCN_type = "dirichlet"
SN = f.def_BCSolution(BCN_type, 10, dy, k, 1)

BCS_type = "dirichlet"
SS = f.def_BCSolution(BCS_type, 10, dy, k, 0)

S = f.enforceBCs(num_nodes, numXNodes, numYNodes, S, SW, SE, SN, SS, BCW_type, BCE_type, BCN_type, BCS_type)

# Build forward difference matrix given BC types 
matrix_i = f.build_matrix(num_nodes, numXNodes, numYNodes, BCW_type, BCE_type, BCN_type, BCS_type, dx, dy)
matrix = np.array(matrix_i)

answer = np.linalg.solve(matrix,S)
'''guess = np.zeros(num_nodes)
answer = f.gauss_seidel_solver(matrix, S, guess)
'''
answer2D = answer.reshape((numYNodes, numXNodes))

plt.figure(figsize=(8,6))
plt.imshow(answer2D, extent=[0, xL, 0, yL], origin='lower', cmap='viridis')
plt.colorbar(label='Temperature')
plt.title('Steady State Heat Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(False)
plt.show()










