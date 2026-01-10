"""
2D steady-state Poisson equation solver for heat conduction.
"""
import numpy as np
from ..utils.finite_difference import build_matrix_2D, def_forceFunction_2D
from ..utils.boundary_conditions import enforceBCs, def_BCSolution


def solve_2d_steady(numXNodes, numYNodes, xL, yL, k, 
                   BCW_type, BCE_type,
                   BCN_type, BCS_type,
                   source_i, source_j, source_strength,
                   SW, SE, SN, SS):
    """
    Solve 2D steady-state heat conduction equation.
    
    Parameters:
    -----------
    numXNodes : int
        Number of nodes in x-direction
    numYNodes : int
        Number of nodes in y-direction
    xL : float
        Domain length in x-direction
    yL : float
        Domain length in y-direction
    k : float
        Thermal conductivity
    BCW_type, BCE_type, BCN_type, BCS_type : str
        Boundary condition types
    source_i, source_j : int
        Source location indices
    source_strength : float
        Source strength
    SW, SE, SN, SS : float
        Boundary values
        Temperature for Dirichlet, Flux for Neumann
        SW : West boundary value
        SE : East boundary value
        SN : North boundary value
        SS : South boundary value
    
    Returns:
    --------
    x, y : ndarray
        Spatial coordinates
    T : ndarray
        Temperature solution (2D array)
    """
    num_nodes = numXNodes * numYNodes
    x = np.linspace(0, xL, numXNodes)
    y = np.linspace(0, yL, numYNodes)
    dx = xL / (numXNodes - 1)
    dy = yL / (numYNodes - 1)
    
    # Build solution vector given BCs and desired source value
    S_vector = def_forceFunction_2D(num_nodes, source_i, source_j, k, source_strength, numXNodes)
    
    SW_vector = def_BCSolution(BCW_type, SW, k)
    SE_vector = def_BCSolution(BCE_type, SE, k)
    SN_vector = def_BCSolution(BCN_type, SN, k)
    SS_vector = def_BCSolution(BCS_type, SS, k)
    
    S = enforceBCs(num_nodes, numXNodes, numYNodes, S_vector, SW_vector, SE_vector, SN_vector, SS_vector, 
                   BCW_type, BCE_type, BCN_type, BCS_type)
    
    # Build forward difference matrix given BC types
    matrix_i = build_matrix_2D(num_nodes, numXNodes, numYNodes, BCW_type, BCE_type, 
                               BCN_type, BCS_type, dx, dy)
    matrix = np.array(matrix_i)
    answer = np.linalg.solve(matrix, S)
    
    answer2D = answer.reshape((numYNodes, numXNodes))
    
    return x, y, answer2D

