"""
2D transient heat conduction solver.
"""
import numpy as np
from ..utils.boundary_conditions import boundary_matrix, def_BCSolution


def initial_condition(num_nodes, numXNodes, numYNodes, T0, SW, SE, SN, SS):
    """
    Create initial condition with boundary values.
    
    Parameters:
    -----------
    num_nodes : int
        Total number of nodes
    numXNodes : int
        Number of nodes in x-direction
    numYNodes : int
        Number of nodes in y-direction
    T0 : float
        Initial temperature
    SW, SE, SN, SS : float
        Boundary values
    
    Returns:
    --------
    initial_condition : ndarray
        Initial temperature field
    """
    initial_condition = np.full(num_nodes, T0)
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


def time_step(num_nodes, numXNodes, numYNodes, S, dt, k, rho, cp, dx, dy, 
              SW, SE, SN, SS, q):
    """
    Perform one time step for 2D transient heat equation.
    
    Parameters:
    -----------
    num_nodes : int
        Total number of nodes
    numXNodes : int
        Number of nodes in x-direction
    numYNodes : int
        Number of nodes in y-direction
    S : ndarray
        Current temperature field
    dt : float
        Time step
    k : float
        Thermal conductivity
    rho : float
        Density
    cp : float
        Specific heat
    dx, dy : float
        Grid spacings
    SW, SE, SN, SS : float
        Boundary values
    q : ndarray
        Heat source array
    
    Returns:
    --------
    S_new : list
        Updated temperature field
    """
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
            deriv = ((S[i+1] + S[i-1] - 2*S[i]) / (dx**2)) + \
                   ((S[i+numXNodes] + S[i-numXNodes] - 2*S[i]) / (dy**2))
            S_new.append(S[i] + alpha * dt * deriv + (dt / (rho * cp)) * q[i])
    return S_new


def transient_run(time, dt, num_nodes, numXNodes, numYNodes, initial_condition, 
                 k, rho, cp, dx, dy, SW, SE, SN, SS, q):
    """
    Run transient 2D heat conduction simulation.
    
    Parameters:
    -----------
    time : float
        Total simulation time
    dt : float
        Time step
    num_nodes : int
        Total number of nodes
    numXNodes : int
        Number of nodes in x-direction
    numYNodes : int
        Number of nodes in y-direction
    initial_condition : ndarray
        Initial temperature field
    k : float
        Thermal conductivity
    rho : float
        Density
    cp : float
        Specific heat
    dx, dy : float
        Grid spacings
    SW, SE, SN, SS : float
        Boundary values
    q : ndarray
        Heat source array
    
    Returns:
    --------
    transient : list
        List of temperature fields at each time step
    """
    num_runs = int(time / dt)
    S_old = initial_condition 
    transient = []
    for i in range(num_runs):
        S_new = time_step(num_nodes, numXNodes, numYNodes, S_old, dt, k, rho, cp, 
                         dx, dy, SW, SE, SN, SS, q)
        transient.append(S_new)
        S_old = S_new
    return transient


def solve_2d_transient(numXNodes, numYNodes, xL, yL, time, dt, k, rho, cp, T0,
                      BCW_type="dirichlet", BCE_type="dirichlet",
                      BCN_type="dirichlet", BCS_type="dirichlet",
                      SW_temp=0, SE_temp=0, SN_temp=0, SS_temp=0,
                      source_i=25, source_j=25, source_strength=10000000):
    """
    Solve 2D transient heat conduction equation.
    
    Parameters:
    -----------
    numXNodes : int
        Number of nodes in x-direction
    numYNodes : int
        Number of nodes in y-direction
    xL, yL : float
        Domain dimensions
    time : float
        Total simulation time
    dt : float
        Time step
    k : float
        Thermal conductivity
    rho : float
        Density
    cp : float
        Specific heat
    T0 : float
        Initial temperature
    BCW_type, BCE_type, BCN_type, BCS_type : str
        Boundary condition types
    SW_temp, SE_temp, SN_temp, SS_temp : float
        Boundary temperatures
    source_i, source_j : int
        Source location indices
    source_strength : float
        Source strength
    
    Returns:
    --------
    x, y : ndarray
        Spatial coordinates
    transient : list
        List of temperature fields at each time step
    """
    num_nodes = numXNodes * numYNodes
    x = np.linspace(0, xL, numXNodes)
    y = np.linspace(0, yL, numYNodes)
    dx = xL / (numXNodes - 1)
    dy = yL / (numYNodes - 1)
    
    # Heat flux array
    q = np.zeros(num_nodes)
    q[numXNodes * source_j + source_i] = source_strength
    
    # Store BCs
    # For Dirichlet: pass temperature directly; for Neumann: pass flux and grid spacing
    SW = def_BCSolution(BCW_type, SW_temp, k, dx if BCW_type == "neumann" else None)
    SE = def_BCSolution(BCE_type, SE_temp, k, dx if BCE_type == "neumann" else None)
    SN = def_BCSolution(BCN_type, SN_temp, k, dy if BCN_type == "neumann" else None)
    SS = def_BCSolution(BCS_type, SS_temp, k, dy if BCS_type == "neumann" else None)
    
    # Build initial condition
    ic = initial_condition(num_nodes, numXNodes, numYNodes, T0, SW, SE, SN, SS)
    
    # Solve
    transient = transient_run(time, dt, num_nodes, numXNodes, numYNodes, ic, 
                             k, rho, cp, dx, dy, SW, SE, SN, SS, q)
    
    return x, y, transient

