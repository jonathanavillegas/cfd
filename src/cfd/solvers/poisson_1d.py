"""
1D steady-state Poisson equation solver for heat conduction.
"""
import numpy as np
from ..utils.finite_difference import build_matrix_1D, def_forceFunction_1D
from ..utils.boundary_conditions import enforceBC_1D


def solve_1d_steady(num_nodes, length, k, BCL_type, BCR_type, sl, sr, 
                   source_pos, source_strength):
    """
    Solve 1D steady-state heat conduction equation.
    
    Parameters:
    -----------
    num_nodes : int
        Number of nodes
    length : float
        Domain length
    k : float
        Thermal conductivity
    BCL_type : str
        Left boundary condition type: "dirichlet" or "neumann"
    BCR_type : str
        Right boundary condition type: "dirichlet" or "neumann"
    sl : float
        Left boundary temperature (Dirichlet)
        or
        Left boundary flux (Neumann)
    sr : float
        Right boundary temperature (Dirichlet)
        or
        Right boundary flux (Neumann)
    source_index : int
        Index of source location
    source_strength : float
        Source strength
    
    Returns:
    --------
    x : ndarray
        Spatial coordinates
    T : ndarray
        Temperature solution
    """
    x = np.linspace(0, length, num_nodes)
    dx = length / (num_nodes - 1)
    
    # Define BCs
    BCL = np.zeros(num_nodes)
    if BCL_type == "dirichlet":
        BCL[0] = 1
        SL = sl
    elif BCL_type == "neumann":
        BCL[0] = -2/dx**2
        BCL[1] = 2/dx**2
        C = sl
        SL =-(C) / k
    else:
        raise ValueError(f"Invalid left boundary condition type: {BCL_type}. Must be 'dirichlet' or 'neumann'.")
    
    BCR = np.zeros(num_nodes)
    if BCR_type == "dirichlet":
        BCR[-1] = 1
        SR = sr
    elif BCR_type == "neumann":
        BCR[-1] = -2/dx**2
        BCR[-2] = 2/dx**2
        C = sr
        SR = -(C) / k
    else:
        raise ValueError(f"Invalid right boundary condition type: {BCR_type}. Must be 'dirichlet' or 'neumann'.")
    
    # Define forcing function
    fx = def_forceFunction_1D(num_nodes, k, length, dx, source_pos, source_strength)
    S = enforceBC_1D(fx, SL, SR)
    
    # Build and solve matrix
    matrix_i = build_matrix_1D(num_nodes, dx, BCL, BCR)
    matrix = np.array(matrix_i)
    answer = np.linalg.solve(matrix, S)
    
    return x, answer

