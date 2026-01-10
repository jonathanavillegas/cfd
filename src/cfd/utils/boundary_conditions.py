"""
Boundary condition utilities for finite difference methods.

Supports Dirichlet, Neumann, and Robin boundary conditions.
"""
import numpy as np


def boundary_matrix(num_nodes, numXNodes, numYNodes):
    """
    Identify boundary node indices for 2D Cartesian grid.
    
    Parameters:
    -----------
    num_nodes : int
        Total number of nodes
    numXNodes : int
        Number of nodes in x-direction (columns)
    numYNodes : int
        Number of nodes in y-direction (rows)
    
    Returns:
    --------
    BW, BE, BN, BS : lists
        Indices of nodes on West, East, North, South boundaries
    """

    # Define points on west and east boundary
    BW = []
    BE = []
    for i in range(numYNodes):
        pointWest = numXNodes * i
        pointEast = numXNodes * i + numXNodes - 1
        BW.append(pointWest)
        BE.append(pointEast)

    # Define points on north and south boundary
    BS = []
    BN = []
    for i in range(numXNodes):
        pointSouth = i
        pointNorth = num_nodes - i - 1
        BS.append(pointSouth)
        BN.append(pointNorth)
    
    return BW, BE, BN, BS


def set_dirichlet(num_nodes, i):
    """
    Create Dirichlet boundary condition row for matrix.
    
    Parameters:
    -----------
    num_nodes : int
        Total number of nodes
    i : int
        Node index where BC is applied
    
    Returns:
    --------
    BC : ndarray
        Matrix row for Dirichlet BC
    """
    BC = np.zeros(num_nodes)
    BC[i] = 1
    return BC


def set_neumann(num_nodes, dx, dy, numXNodes, i):
    """
    Create Neumann boundary condition row for matrix.
    
    Parameters:
    -----------
    num_nodes : int
        Total number of nodes
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
    numXNodes : int
        Number of nodes in x-direction
    i : int
        Node index where BC is applied
    
    Returns:
    --------
    BC : ndarray
        Matrix row for Neumann BC
    """
    BC = np.zeros(num_nodes)
    # Center node
    BC[i] = (-2/dx**2) + (-2/dy**2)
    # East node
    BC[i+1] = 2/dx**2
    # No west node bc ghost
    # South node
    BC[i-numXNodes] = 1/dy**2
    # North node
    BC[i+numXNodes] = 1/dy**2
    return BC


def def_BCSolution(BCtype, S, k):
    """
    Calculate solution value for boundary condition.
    
    Parameters:
    -----------
    BCtype : str
        Type of BC: "dirichlet", or "neumann"
    S : float
        Source value
    k : float
        Thermal conductivity
    
    Returns:
    --------
    S : float
        Solution value for boundary
    """
    if BCtype == "dirichlet":
        return S
    elif BCtype == "neumann":
        return (-S) / k
    else:
        raise ValueError(f"Unknown boundary condition type: {BCtype}")


def enforceBCs(num_nodes, numXNodes, numYNodes, S, SW, SE, SN, SS, 
               BCW_type, BCE_type, BCN_type, BCS_type):
    """
    Enforce boundary conditions on solution vector.
    
    Parameters:
    -----------
    num_nodes : int
        Total number of nodes
    numXNodes : int
        Number of nodes in x-direction
    numYNodes : int
        Number of nodes in y-direction
    S : ndarray
        Solution vector
    SW, SE, SN, SS : float
        Solution values for West, East, North, South boundaries
    BCW_type, BCE_type, BCN_type, BCS_type : str
        Boundary condition types for each boundary
    
    Returns:
    --------
    S : ndarray
        Solution vector with BCs enforced
    """
    BW, BE, BN, BS = boundary_matrix(num_nodes, numXNodes, numYNodes)

    for point in BW:
        S[point] = SW
    for point in BE:
        S[point] = SE
    for point in BS:
        S[point] = SS
    for point in BN:
        S[point] = SN   
    
    # Overwrite corners
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


def enforceBC_1D(fx, SL, SR):
    """
    Enforce boundary conditions for 1D problem.
    
    Parameters:
    -----------
    fx : ndarray
        Force/source vector
    SL : float
        Left boundary value
    SR : float
        Right boundary value
    
    Returns:
    --------
    S : ndarray
        Solution vector with BCs enforced
    """
    S = fx.copy()
    S[0] = SL
    S[-1] = SR
    return S

