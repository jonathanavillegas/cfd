"""
Finite difference matrix building and stencil utilities.
"""
import numpy as np
from .boundary_conditions import boundary_matrix, set_dirichlet, set_neumann


def build_matrix_1D(num_nodes, dx, BCL, BCR):
    """
    Build finite difference matrix for 1D Poisson equation.
    
    Parameters:
    -----------
    num_nodes : int
        Number of nodes
    dx : float
        Grid spacing
    BCL : ndarray
        Left boundary condition row
    BCR : ndarray
        Right boundary condition row
    
    Returns:
    --------
    matrix_i : list
        List of matrix rows
    """
    matrix_i = []
    # Build each row and append to matrix; first and last will have 1s for Dirichlet BCs
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
    return matrix_i


def build_matrix_2D(num_nodes, numXNodes, numYNodes, BCW_type, BCE_type, 
                    BCN_type, BCS_type, dx, dy):
    """
    Build finite difference matrix for 2D Poisson equation.
    
    Parameters:
    -----------
    num_nodes : int
        Total number of nodes
    numXNodes : int
        Number of nodes in x-direction
    numYNodes : int
        Number of nodes in y-direction
    BCW_type, BCE_type, BCN_type, BCS_type : str
        Boundary condition types
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
    
    Returns:
    --------
    matrix_i : list
        List of matrix rows
    """
    BW, BE, BN, BS = boundary_matrix(num_nodes, numXNodes, numYNodes)
    
    matrix_i = []
    # Build each row and append to matrix; if at boundary then set corresponding to BC type
    for i in range(num_nodes):
        eq = np.zeros(num_nodes)

        # Sets top left corner
        if i in BW and i in BN:
            if BCW_type == "dirichlet" or BCN_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif (BCW_type == "neumann" or BCW_type == "nuemann") and \
                 (BCN_type == "neumann" or BCN_type == "nuemann"):
                BCNW = np.zeros(num_nodes)
                # Center node
                BCNW[i] = (-2/dx**2) + (-2/dy**2)
                # East node
                BCNW[i+1] = 2/dx**2
                # No west node bc ghost
                # South node
                BCNW[i-numXNodes] = 2/dy**2
                # No north node bc ghost
                eq = BCNW

        # Set bottom left corner
        elif i in BW and i in BS:
            if BCW_type == "dirichlet" or BCS_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif (BCW_type == "neumann" or BCW_type == "nuemann") and \
                 (BCS_type == "neumann" or BCS_type == "nuemann"):
                BCSW = np.zeros(num_nodes)
                # Center node
                BCSW[i] = (-2/dx**2) + (-2/dy**2)
                # East node
                BCSW[i+1] = 2/dx**2
                # No west node bc ghost
                # South node
                BCSW[i-numXNodes] = 2/dy**2
                # No north node bc ghost
                eq = BCSW

        # Sets top right corner
        elif i in BE and i in BN:
            if BCE_type == "dirichlet" or BCN_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif (BCE_type == "neumann" or BCE_type == "nuemann") and \
                 (BCN_type == "neumann" or BCN_type == "nuemann"):
                BCNE = np.zeros(num_nodes)
                # Center node
                BCNE[i] = (-2/dx**2) + (-2/dy**2)
                # East node
                BCNE[i+1] = 2/dx**2
                # No west node bc ghost
                # South node
                BCNE[i-numXNodes] = 2/dy**2
                # No north node bc ghost
                eq = BCNE
        
        # Set bottom right corner
        elif i in BE and i in BS:
            if BCE_type == "dirichlet" or BCS_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif (BCE_type == "neumann" or BCE_type == "nuemann") and \
                 (BCS_type == "neumann" or BCS_type == "nuemann"):
                BCSE = np.zeros(num_nodes)
                # Center node
                BCSE[i] = (-2/dx**2) + (-2/dy**2)
                # East node
                BCSE[i+1] = 2/dx**2
                # No west node bc ghost
                # South node
                BCSE[i-numXNodes] = 2/dy**2
                # No north node bc ghost
                eq = BCSE
                
        # Sets left boundary
        elif i in BW:
            if BCW_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCW_type == "neumann" or BCW_type == "nuemann":
                BCW = set_neumann(num_nodes, dx, dy, numXNodes, i)
                eq = BCW
        # Set right boundary
        elif i in BE:
            if BCE_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCE_type == "neumann" or BCE_type == "nuemann":
                BCE = np.zeros(num_nodes)
                # Center node
                BCE[i] = (-2/dx**2) + (-2/dy**2)
                # No east node because ghost
                # West node
                BCE[i-1] = 2/dx**2
                # South node
                BCE[i-numXNodes] = 1/dy**2
                # North node
                BCE[i+numXNodes] = 1/dy**2
                eq = BCE
        # Set top boundary
        elif i in BN:
            if BCN_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCN_type == "neumann" or BCN_type == "nuemann":
                BCN = np.zeros(num_nodes)
                # Center node
                BCN[i] = (-2/dx**2) + (-2/dy**2)
                # East node
                BCN[i+1] = 1/dx**2
                # West node
                BCN[i-1] = 1/dx**2
                # South node
                BCN[i-numXNodes] = 2/dy**2
                # No north node because ghost
                eq = BCN
        # Set bottom boundary
        elif i in BS:
            if BCS_type == "dirichlet":
                eq = set_dirichlet(num_nodes, i)
            elif BCS_type == "neumann" or BCS_type == "nuemann":
                BCS = np.zeros(num_nodes)
                # Center node
                BCS[i] = (-2/dx**2) + (-2/dy**2)
                # No east node because ghost
                BCS[i+1] = 1/dx**2
                # West node
                BCS[i-1] = 1/dx**2
                # No south node because ghost
                # North node
                BCS[i+numXNodes] = 2/dy**2
                eq = BCS
        # Middle nodes
        else:
            # Center node
            eq[i] = (-2/dx**2) + (-2/dy**2)
            # East node
            eq[i+1] = 1/dx**2
            # West node
            eq[i-1] = 1/dx**2
            # South node
            eq[i-numXNodes] = 1/dy**2
            # North node
            eq[i+numXNodes] = 1/dy**2

        matrix_i.append(eq)
    return matrix_i


def def_forceFunction_1D(num_nodes, k, length, dx, source_pos, source_strength):
    """
    Define forcing function for 1D problem.
    
    Parameters:
    -----------
    num_nodes : int
        Number of nodes
    k : float
        Thermal conductivity
    length : float
        Length of domain
    source_pos : int
        Position of source location (m from left boundary)
    source_strength : float
        Source strength
    
    Returns:
    --------
    f : ndarray
        Force vector
    """
    f = np.zeros(num_nodes)
    # Add source here
    f[int(source_pos/length * num_nodes)] = -source_strength / dx
    f = np.array(f)
    f = f/k
    return f


def def_forceFunction_2D(num_nodes, source_i, source_j, k, S, numXNodes, numYNodes, xL, yL):
    """
    Define forcing function for 2D problem.
    
    Parameters:
    -----------
    num_nodes : int
        Total number of nodes
    source_i : int
        x-index of source location
    source_j : int
        y-index of source location
    k : float
        Thermal conductivity
    S : float
        Source strength (volumetric source)
    numXNodes : int
        Number of nodes in x-direction
    numYNodes : int
        Number of nodes in y-direction
    xL : float
        Length of domain in x-direction
    yL : float
        Length of domain in y-direction
    
    Returns:
    --------
    f : ndarray
        Force vector
    """
    dx = xL / (numXNodes - 1)
    dy = yL / (numYNodes - 1)
    f = np.zeros(num_nodes)
    # Add sources here
    x_pos = source_i * numXNodes / xL
    y_pos = source_j * numYNodes / yL
    XY = (int(x_pos) + (int(y_pos) - 1) * numXNodes)
    # Ensure index is within bounds before assigning
    if 0 <= XY < num_nodes:
        f[XY] = -S / (dx * dy)
    f = np.array(f)
    f = f/k
    return f

