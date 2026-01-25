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
              BCW_type, BCE_type, BCN_type, BCS_type,
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
    BCW_type, BCE_type, BCN_type, BCS_type : str
        Boundary condition types
    SW, SE, SN, SS : float
        Boundary values (temperature for Dirichlet, flux for Neumann)
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
        # West boundary
        if i in BW:
            if BCW_type == "dirichlet":
                S_new.append(SW)
            elif BCW_type == "neumann" or BCW_type == "nuemann":
                # Neumann BC: positive flux = heat in, negative = heat out
                if i + 1 < num_nodes:
                    S_new.append(S[i+1] + SW*dx/k)
                else:
                    S_new.append(SW)
            else:
                S_new.append(SW)
        # East boundary
        elif i in BE:
            if BCE_type == "dirichlet":
                S_new.append(SE)
            elif BCE_type == "neumann" or BCE_type == "nuemann":
                # Neumann BC: positive flux = heat in, negative = heat out
                if i - 1 >= 0:
                    S_new.append(S[i-1] + SE*dx/k)
                else:
                    S_new.append(SE)
            else:
                S_new.append(SE)
        # North boundary
        elif i in BN:
            if BCN_type == "dirichlet":
                S_new.append(SN)
            elif BCN_type == "neumann" or BCN_type == "nuemann":
                # Neumann BC: positive flux = heat in, negative = heat out
                if i - numXNodes >= 0:
                    S_new.append(S[i-numXNodes] + SN*dy/k)
                else:
                    S_new.append(SN)
            else:
                S_new.append(SN)
        # South boundary
        elif i in BS:
            if BCS_type == "dirichlet":
                S_new.append(SS)
            elif BCS_type == "neumann" or BCS_type == "nuemann":
                # Neumann BC: positive flux = heat in, negative = heat out
                if i + numXNodes < num_nodes:
                    S_new.append(S[i+numXNodes] + SS*dy/k)
                else:
                    S_new.append(SS)
            else:
                S_new.append(SS)
        # Interior nodes
        else:
            deriv = ((S[i+1] + S[i-1] - 2*S[i]) / (dx**2)) + \
                   ((S[i+numXNodes] + S[i-numXNodes] - 2*S[i]) / (dy**2))
            S_new.append(S[i] + alpha * dt * deriv + (dt / (rho * cp)) * q[i])
    return S_new


def transient_run(time, dt, num_nodes, numXNodes, numYNodes, initial_condition, 
                 k, rho, cp, dx, dy, BCW_type, BCE_type, BCN_type, BCS_type,
                 SW, SE, SN, SS, q):
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
    BCW_type, BCE_type, BCN_type, BCS_type : str
        Boundary condition types
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
                         dx, dy, BCW_type, BCE_type, BCN_type, BCS_type,
                         SW, SE, SN, SS, q)
        transient.append(S_new)
        S_old = S_new
    return transient


def solve_2d_transient(numXNodes, numYNodes, xL, yL, time, dt, k, rho, cp, T0,
                      BCW_type, BCE_type,
                      BCN_type, BCS_type,
                      SW, SE, SN, SS,
                      source_x, source_y, source_strength):
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
    SW, SE, SN, SS : float
        Boundary temperatures
    source_x, source_y : float
        Source location coordinates (x, y values in domain)
        Must be within [0, xL] and [0, yL] respectively
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
    
    # Convert source coordinates to indices and validate bounds
    if source_x is not None and source_y is not None and source_strength != 0:
        # Check if coordinates are within bounds
        if source_x < 0 or source_x > xL:
            raise ValueError(f"Source x-coordinate {source_x} is outside domain bounds [0, {xL}]")
        if source_y < 0 or source_y > yL:
            raise ValueError(f"Source y-coordinate {source_y} is outside domain bounds [0, {yL}]")
        
        # Convert coordinates to indices
        # Find closest node indices
        source_i = int(np.round(source_x / xL * (numXNodes - 1)))
        source_j = int(np.round(source_y / yL * (numYNodes - 1)))
        
        # Ensure indices are within valid range
        source_i = max(0, min(source_i, numXNodes - 1))
        source_j = max(0, min(source_j, numYNodes - 1))
        
        # Convert to linear index
        linear_index = source_j * numXNodes + source_i
        q[linear_index] = source_strength
    
    # Process BCs: For Dirichlet, use temperature directly; for Neumann, use flux directly
    # For initial condition, use temperature values (convert Neumann flux if needed)
    SW_init = SW if BCW_type == "dirichlet" else T0  # For Neumann, start with T0
    SE_init = SE if BCE_type == "dirichlet" else T0
    SN_init = SN if BCN_type == "dirichlet" else T0
    SS_init = SS if BCS_type == "dirichlet" else T0
    
    # Build initial condition
    ic = initial_condition(num_nodes, numXNodes, numYNodes, T0, SW_init, SE_init, SN_init, SS_init)
    
    # For time stepping, pass BC values as-is:
    # - Dirichlet: temperature value
    # - Neumann: flux value (will be used in time_step)
    # Solve
    transient = transient_run(time, dt, num_nodes, numXNodes, numYNodes, ic, 
                             k, rho, cp, dx, dy, BCW_type, BCE_type, BCN_type, BCS_type,
                             SW, SE, SN, SS, q)
    
    return x, y, transient

