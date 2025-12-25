#solver should be able to solve for Poisson EQ (steady state heat with constant heat transfer coefficient)
import numpy as np
import matplotlib.pyplot as plt

#defines physics forcing function
def def_forceFunction(num_nodes,k):
    f = np.zeros(num_nodes)

    #add sources here
    f[50] = -100
    f = np.array(f)
    f = f/k
    return(f)

#enforces BC
def enforceBC(fx, SL, SR):
    S = fx
    S[0] = SL
    S[-1] = SR
    return S

#builds finite difference matrix 
def build_matrix(num_nodes, dx, BCL, BCR):
    matrix_i = []
    #build each row and append to matrix; first and last will have 1s for Dirichlet BCs
    for i in range(num_nodes):
        eq = np.zeros(num_nodes)
        if i == 0:
            eq = BCL
        elif i == num_nodes - 1:
            eq = BCR
        else:
            eq[i-1] = 1/dx**2
            eq[i] = -2/dx**2
            eq[i+1] = 1/dx**2
        matrix_i.append(eq)
    return(matrix_i)


#define 1D geometry
num_nodes = 100
length = 10 
x = np.linspace(0,length, num_nodes)
dx = length / (num_nodes - 1)
k = 5
BCR_type = "neumann"

#define BC; Dirichlet for now; q0 near zero bc need actual value to solve matrix

qR = 10
#Left boundary will always be Dirichlet for now
BCL = np.zeros(num_nodes)
BCL[0] = 1
SL = .0001
#Set Dirichlet, Neumann, or Robin BCs at right boundary
BCR = np.zeros(num_nodes)
if BCR_type == "dirichlet":
    BCR[-1] = 1
    #constant temp value
    SR = 10
elif BCR_type == "neumann":
    BCR[-1] = -2/dx**2
    BCR[-2] = 2/dx**2
    #flux term
    C = 1
    #source term 
    S = 0
    SR = -(S - C/dx) / k
elif BCR_type == "robin":
    #mixed BC coefficients a*(dT/dx) + b*T(x) = c
    a = 100
    b = 10
    c = 10
    BCR[-1] = (2/dx**2) * (1 + b*dx/a)
    #source term
    S = 0
    SR = -(S - (2*c)/(dx*a)) / k

#define EQ; we seek to find 2nd derivative using forward difference
fx = def_forceFunction(num_nodes, k)
S = enforceBC(fx, SL, SR)

matrix_i = build_matrix(num_nodes, dx, BCL, BCR)
matrix = np.array(matrix_i)
answer = np.linalg.solve(matrix,S)

plt.plot(x,answer)
plt.show()









