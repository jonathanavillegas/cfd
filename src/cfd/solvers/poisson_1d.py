"""
1D steady-state Poisson equation solver for heat conduction.
"""
import numpy as np
from ..utils.finite_difference import build_matrix_1D, def_forceFunction_1D
from ..utils.boundary_conditions import enforceBC_1D


def solve_1d_steady(num_nodes, length, k, BCR_type="neumann", qR=10, SL=0.0001, 
                   source_index=50, source_strength=-100):
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
    BCR_type : str
        Right boundary condition type: "dirichlet", "neumann", or "robin"
    qR : float
        Right boundary flux (for Neumann)
    SL : float
        Left boundary temperature (Dirichlet)
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
    BCL[0] = 1
    
    BCR = np.zeros(num_nodes)
    if BCR_type == "dirichlet":
        BCR[-1] = 1
        SR = 10
    elif BCR_type == "neumann":
        BCR[-1] = -2/dx**2
        BCR[-2] = 2/dx**2
        C = 1
        S = 0
        SR = -(S - C/dx) / k
    elif BCR_type == "robin":
        a = 100
        b = 10
        c = 10
        BCR[-1] = (2/dx**2) * (1 + b*dx/a)
        S = 0
        SR = -(S - (2*c)/(dx*a)) / k
    
    # Define forcing function
    fx = def_forceFunction_1D(num_nodes, k)
    S = enforceBC_1D(fx, SL, SR)
    
    # Build and solve matrix
    matrix_i = build_matrix_1D(num_nodes, dx, BCL, BCR)
    matrix = np.array(matrix_i)
    answer = np.linalg.solve(matrix, S)
    
    return x, answer

