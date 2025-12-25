#helper functions 
import numpy as np
from scipy.interpolate import interp1d


def boundary_matrix(num_nodes, numXNodes, numYNodes):
    BW = []
    BE = []
    for i in range(numYNodes):
        pointWest = numXNodes * i
        pointEast = numXNodes * i + numXNodes - 1
        BW.append(pointWest)
        BE.append(pointEast)

    #define points on north and south boundary
    BS = []
    BN = []
    for i in range(numXNodes):
        pointSouth = i
        pointNorth = num_nodes - i - 1
        BS.append(pointSouth)
        BN.append(pointNorth)
    
    return BW, BE, BN, BS

def set_dirichlet(num_nodes, i):
    BC = np.zeros(num_nodes)
    BC[i] = 1
    return BC

def set_nuemann(num_nodes, dx, dy, numXNodes, i):
    BC = np.zeros(num_nodes)
    #center node
    BC[i] = (-2/dx**2) + (-2/dy**2)
    #east node
    BC[i+1] = 2/dx**2
    #no west node bc ghost
    #south node
    BC[i-numXNodes] = 1/dy**2
    #north node
    BC[i+numXNodes] = 1/dy**2
    return BC

#builds finite difference matrix 
#### need to fix corners 
def build_matrix(num_nodes, numXNodes, numYNodes, BCW_type, BCE_type, BCN_type, BCS_type, dx, dy):
    BW, BE, BN, BS = boundary_matrix(num_nodes, numXNodes, numYNodes)
    
    matrix_i = []
    #build each row and append to matrix; if at boundary then set corresponding to BC type
    for i in range(num_nodes):
        eq = np.zeros(num_nodes)

        #sets top left corner
        if i in BW and i in BN:
            if BCW_type == "dirichlet" or BCN_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCW_type == "nuemann" and BCN_type == "nuemann":
                BCNW = np.zeros(num_nodes)
                #center node
                BCNW[i] = (-2/dx**2) + (-2/dy**2)
                #east node
                BCNW[i+1] = 2/dx**2
                #no west node bc ghost
                #south node
                BCNW[i-numXNodes] = 2/dy**2
                #no north node bc ghost
                eq = BCNW

        #set bottom left corner
        elif i in BW and i in BS:
            if BCW_type == "dirichlet" or BCS_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCW_type == "nuemann" and BCS_type == "nuemann":
                BCSW = np.zeros(num_nodes)
                #center node
                BCSW[i] = (-2/dx**2) + (-2/dy**2)
                #east node
                BCSW[i+1] = 2/dx**2
                #no west node bc ghost
                #south node
                BCSW[i-numXNodes] = 2/dy**2
                #no north node bc ghost
                eq = BCSW

        #sets top right corner
        elif i in BE and i in BN:
            if BCE_type == "dirichlet" or BCN_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCE_type == "nuemann" and BCN_type == "nuemann":
                BCNE = np.zeros(num_nodes)
                #center node
                BCNE[i] = (-2/dx**2) + (-2/dy**2)
                #east node
                BCNE[i+1] = 2/dx**2
                #no west node bc ghost
                #south node
                BCNE[i-numXNodes] = 2/dy**2
                #no north node bc ghost
                eq = BCNE
        
        #set bottom right corner
        elif i in BE and i in BS:
            if BCE_type == "dirichlet" or BCS_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCE_type == "nuemann" and BCS_type == "nuemann":
                BCSE = np.zeros(num_nodes)
                #center node
                BCSE[i] = (-2/dx**2) + (-2/dy**2)
                #east node
                BCSE[i+1] = 2/dx**2
                #no west node bc ghost
                #south node
                BCSE[i-numXNodes] = 2/dy**2
                #no north node bc ghost
                eq = BCSE
                
        #sets left boundary
        elif i in BW:
            if BCW_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCW_type == "nuemann":
                BCW = np.zeros(num_nodes)
                #center node
                BCW[i] = (-2/dx**2) + (-2/dy**2)
                #east node
                BCW[i+1] = 2/dx**2
                #no west node bc ghost
                #south node
                BCW[i-numXNodes] = 1/dy**2
                #north node
                BCW[i+numXNodes] = 1/dy**2
                eq = BCW
        #set right boundary
        elif i in BE:
            if BCE_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCE_type == "nuemann":
                BCE = np.zeros(num_nodes)
                #center node
                BCE[i] = (-2/dx**2) + (-2/dy**2)
                #no east node because ghost
                #west node
                BCE[i-1] = 2/dx**2
                #south node
                BCE[i-numXNodes] = 1/dy**2
                #north node
                BCE[i+numXNodes] = 1/dy**2
                eq = BCE
        #set top boundary
        elif i in BN:
            if BCN_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCN_type == "nuemann":
                BCN = np.zeros(num_nodes)
                #center node
                BCN[i] = (-2/dx**2) + (-2/dy**2)
                #east node
                BCN[i+1] = 1/dx**2
                #west node
                BCN[i-1] = 1/dx**2
                #south node
                BCN[i-numXNodes] = 2/dy**2
                #no north node because ghost
                eq = BCN
        #set bottom boundary
        elif i in BS:
            if BCS_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCS_type == "nuemann":
                BCS = np.zeros(num_nodes)
                #center node
                BCS[i] = (-2/dx**2) + (-2/dy**2)
                #no east node because ghost
                BCS[i+1] = 1/dx**2
                #west node
                BCS[i-1] = 1/dx**2
                #no south node because ghost
                #north node
                BCS[i+numXNodes] = 2/dy**2
                eq = BCS
        #middle nodes
        else:
            #center node
            eq[i] = (-2/dx**2) + (-2/dy**2)
            #east node
            eq[i+1] = 1/dx**2
            #west node
            eq[i-1] = 1/dx**2
            #south node
            eq[i-numXNodes] = 1/dy**2
            #north node
            eq[i+numXNodes] = 1/dy**2

        matrix_i.append(eq)
    return(matrix_i)

#defines physics forcing function
def def_forceFunction(num_nodes, source_i, source_j, k, S, numXNodes):
    f = np.zeros(num_nodes)
    #add sources here
    XY = (source_i + (source_j - 1) * numXNodes)
    f[XY] = -S
    f = np.array(f)
    f = f/k
    return(f)

#define BC array; find points in array that need BC applied and applies
#num_nodes = total number of nodes
#numXNodes = number of columns of nodes
#numYNodes = number of rows of nodes
#S = solution array
#SW = solution value for West BC
#SE = solution value for East BC
#SN = solution value for North BC
#SS = solution value for South BC
def enforceBCs(num_nodes, numXNodes, numYNodes, S, SW, SE, SN, SS, BCW_type, BCE_type, BCN_type, BCS_type):
    #define points on west and east boundary
    BW, BE, BN, BS = boundary_matrix(num_nodes, numXNodes, numYNodes)

    for point in BW:
        S[point] = SW
    for point in BE:
        S[point] = SE
    for point in BS:
        S[point] = SS
    for point in BN:
        S[point] = SN   
    
    #overwrite corners
    for point in range(num_nodes):
        if point in BW and point in BN:
            if BCW_type == "dirichlet" and BCN_type == "dirichlet":
                S[point] = (SW + SN) / 2
            elif BCW_type == "dirichlet":
                S[point] = SW
            elif BCN_type == "dirichlet":
                S[point] = SN
            else:
                S[point] = (SW + SN) / 2  
        elif point in BW and point in BS:
            if BCW_type == "dirichlet" and BCS_type == "dirichlet":
                S[point] = (SW + SS) / 2
            elif BCW_type == "dirichlet":
                S[point] = SW
            elif BCS_type == "dirichlet":
                S[point] = SS
            else:
                S[point] = (SW + SS) / 2
        elif point in BE and point in BN:
            if BCE_type == "dirichlet" and BCN_type == "dirichlet":
                S[point] = (SE + SN) / 2
            elif BCE_type == "dirichlet":
                S[point] = SE
            elif BCN_type == "dirichlet":
                S[point] = SN
            else:
                S[point] = (SE + SN) / 2
        elif point in BE and point in BS:
            if BCE_type == "dirichlet" and BCS_type == "dirichlet":
                S[point] = (SE + SS) / 2
            elif BCE_type == "dirichlet":
                S[point] = SE
            elif BCS_type == "dirichlet":
                S[point] = SS
            else:
                S[point] = (SE + SS) / 2
    return S

#finds the solution value given type of BC, source term, and flux term
#BCtype = type of BC
#S = source value
#C = flux value (if Nuemann or Robin)
#k = thermal conductivity
#d = distance
def def_BCSolution(BCtype, S, d, k, C):
    if BCtype == "dirichlet":
        S = S
    elif BCtype == "nuemann":
        S = (-S + 2*C/d) / k
    elif BCtype == "robin":
        c = 3
        S = (-S + (2*c)/(d)) / k   
    return S


def jacobi_solver(A, b, x0, tol=1, max_iter=1000):
    """
    Solves Ax = b using the Jacobi iterative method.
    
    Parameters:
        A (ndarray): Coefficient matrix (n x n)
        b (ndarray): Right-hand side vector (n,)
        x0 (ndarray): Initial guess vector (n,)
        tol (float): Tolerance for convergence (default: 1e-10)
        max_iter (int): Maximum number of iterations (default: 100)
        
    Returns:
        x (ndarray): Approximate solution vector
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)
    n = len(b)

    for iteration in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum_ax = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_ax) / A[i, i]
        
        # Check convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Converged in {iteration + 1} iterations.")
            return x_new
        print(np.linalg.norm(x_new - x, ord=np.inf))
        x = x_new
    print("Maximum iterations reached without convergence.")
    return x


def gauss_seidel_solver(A, b, x0=None, tol=1e-4, max_iter=1000):

    """
    Solves the linear system Ax = b using the Gauss-Seidel iterative method.

    Parameters:
    - A: Coefficient matrix (must be square, preferably diagonally dominant or SPD)
    - b: Right-hand side vector
    - x0: Initial guess (default: zeros)
    - tol: Tolerance for convergence (default: 1e-10)
    - max_iter: Maximum number of iterations (default: 100)

    Returns:
    - x: Solution vector
    """

    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    for iteration in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])  # updated values
            sum2 = np.dot(A[i, i+1:], x[i+1:])  # old values
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Converged in {iteration+1} iterations.")
            return x_new
        print(np.linalg.norm(x_new - x, ord=np.inf))
        x = x_new

    print("Did not converge within the maximum number of iterations.")
    return x


def initial_condition(num_nodes, numXNodes, numYNodes, T0, SW, SE, SN, SS):
    initial_condition = np.full(num_nodes,T0)
    BW, BE, BN, BS = boundary_matrix(num_nodes, numXNodes, numYNodes)

    for i in range(len(initial_condition)):
        if i in BW:
            initial_condition[i] = SW
        elif i in BE:
            initial_condition[i] = SE
        elif i in BN:
            initial_condition[i] = SN
        elif i in BS:
            initial_condition[i] = SS
    return initial_condition


def time_step(num_nodes, numXNodes, numYNodes, S, dt, k, rho, cp, dx, dy, SW, SE, SN, SS, q):
    alpha = k / (rho * cp)
    BW, BE, BN, BS = boundary_matrix(num_nodes, numXNodes, numYNodes)
    S_new = []
    for i in range(num_nodes):
        if i in BW:
            S_new.append(SW)
        elif i in BE:
            S_new.append(SE)
        elif i in BN:
            S_new.append(SN)
        elif i in BS:
            S_new.append(SS)
        else:
            deriv = ((S[i+1] + S[i-1] - 2*S[i]) / (dx**2)) + ((S[i+numXNodes] + S[i-numXNodes] - 2*S[i]) / (dy**2))
            S_new.append(S[i] + alpha * dt * deriv + (dt / (rho * cp)) * q[i])
    return S_new


def transient_run(time, dt, num_nodes, numXNodes, numYNodes, initial_condition, k, rho, cp, dx, dy, SW, SE, SN, SS, q):
    num_runs = int(time / dt)
    S_old = initial_condition 
    transient = []
    for i in range(num_runs):
        S_new = time_step(num_nodes, numXNodes, numYNodes, S_old, dt, k, rho, cp, dx, dy, SW, SE, SN, SS, q)
        transient.append(S_new)
        S_old = S_new
    return transient





