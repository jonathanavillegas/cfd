"""
2D steady-state Poisson equation solver for heat conduction.
"""
import numpy as np
from ..utils.finite_difference import build_matrix_2D, def_forceFunction_2D
from ..utils.boundary_conditions import enforceBCs, def_BCSolution


def solve_2d_steady(numXNodes, numYNodes, xL, yL, k, 
                   BCW_type="neumann", BCE_type="neumann",
                   BCN_type="dirichlet", BCS_type="dirichlet",
                   source_i=25, source_j=25, source_strength=1000,
                   SW_flux=5, SE_flux=-5, SN_temp=1, SS_temp=0):
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
    SW_flux, SE_flux : float
        West and East boundary fluxes
    SN_temp, SS_temp : float
        North and South boundary temperatures
    
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
    S = def_forceFunction_2D(num_nodes, source_i, source_j, k, source_strength, numXNodes)
    
    SW = def_BCSolution(BCW_type, 0, dx, k, SW_flux)
    SE = def_BCSolution(BCE_type, 0, dx, k, SE_flux)
    SN = def_BCSolution(BCN_type, 10, dy, k, SN_temp)
    SS = def_BCSolution(BCS_type, 10, dy, k, SS_temp)
    
    S = enforceBCs(num_nodes, numXNodes, numYNodes, S, SW, SE, SN, SS, 
                   BCW_type, BCE_type, BCN_type, BCS_type)
    
    # Build forward difference matrix given BC types
    matrix_i = build_matrix_2D(num_nodes, numXNodes, numYNodes, BCW_type, BCE_type, 
                               BCN_type, BCS_type, dx, dy)
    matrix = np.array(matrix_i)
    answer = np.linalg.solve(matrix, S)
    
    answer2D = answer.reshape((numYNodes, numXNodes))
    
    return x, y, answer2D

