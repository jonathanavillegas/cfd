"""
Heat shield transient solver for 2D and 3D cylindrical coordinates.

Note: Ghost nodes implementation will be added in subsequent tasks.
"""
import numpy as np
from ..utils.flux_distribution import create_flux_distribution


def time_step_2D(Nr, Ntheta, dr, dtheta, dt, T, q, r, k, rho, cp):
    """
    Perform one time step for 2D heat shield (r-θ plane) using ghost nodes.
    
    Boundary Conditions:
    - Radial boundary (r=R): Insulated (zero flux)
    - Center (r=0): Handled with L'Hôpital's rule
    - Heat flux q applied as source term (representing top surface flux)
    
    Parameters:
    -----------
    Nr : int
        Number of radial nodes
    Ntheta : int
        Number of angular nodes
    dr : float
        Radial grid spacing
    dtheta : float
        Angular grid spacing
    dt : float
        Time step
    T : ndarray
        Current temperature field (Nr, Ntheta)
    q : ndarray
        Heat flux distribution (Nr, Ntheta) [kW/m²]
    r : ndarray
        Radial coordinates
    k : float
        Thermal conductivity
    rho : float
        Density
    cp : float
        Specific heat
    
    Returns:
    --------
    T_new : ndarray
        Updated temperature field
    """
    T_old = T.copy()
    alpha = k / (rho * cp)
    
    # Create extended array with ghost nodes: (Nr+1, Ntheta)
    # Interior: T_ext[1:Nr+1, :] = T
    # Ghost nodes: T_ext[0, :] (r=0), T_ext[Nr, :] (r=R)
    T_ext = np.zeros((Nr+1, Ntheta))
    T_ext[1:Nr+1, :] = T_old
    
    # Set radial boundary ghost node (r=R) - insulated
    for j in range(Ntheta):
        T_ext[Nr, j] = T_ext[Nr-1, j]
    
    T_new = np.zeros((Nr, Ntheta))
    
    for i in range(Nr):
        for j in range(Ntheta):
            # Map to extended array index
            i_ext = i + 1
            
            # Handle r=0 singularity using L'Hôpital's rule
            if i == 0:
                # At r=0: (1/r)*∂T/∂r → ∂²T/∂r² as r→0
                # Modified radial term: 2 * (T[1] - T[0]) / dr²
                radial_term = 2 * (T_ext[i_ext+1, j] - T_ext[i_ext, j]) / dr**2
                # Angular term vanishes by symmetry at r=0
                angular_term = 0
            elif i == Nr - 1:
                # At radial boundary (r=R), use ghost node
                # Standard radial derivatives using ghost node
                radial_term = ((T_ext[Nr, j] - 2*T_ext[i_ext, j] + T_ext[i_ext-1, j]) / dr**2 +
                              (1/r[i])*(T_ext[Nr, j] - T_ext[i_ext-1, j]) / (2*dr))
                
                # Angular derivatives
                if j == Ntheta - 1:
                    angular_term = (1/r[i]**2) * (T_ext[i_ext, 0] - 2*T_ext[i_ext, j] + T_ext[i_ext, j-1]) / dtheta**2
                else:
                    angular_term = (1/r[i]**2) * (T_ext[i_ext, j+1] - 2*T_ext[i_ext, j] + T_ext[i_ext, j-1]) / dtheta**2
            else:
                # Standard radial derivatives
                radial_term = ((T_ext[i_ext+1, j] - 2*T_ext[i_ext, j] + T_ext[i_ext-1, j]) / dr**2 +
                              (1/r[i])*(T_ext[i_ext+1, j] - T_ext[i_ext-1, j]) / (2*dr))
                
                # Angular derivatives
                if j == Ntheta - 1:
                    angular_term = (1/r[i]**2) * (T_ext[i_ext, 0] - 2*T_ext[i_ext, j] + T_ext[i_ext, j-1]) / dtheta**2
                else:
                    angular_term = (1/r[i]**2) * (T_ext[i_ext, j+1] - 2*T_ext[i_ext, j] + T_ext[i_ext, j-1]) / dtheta**2
            
            # Compute derivative
            deriv = alpha * (radial_term + angular_term) + (q[i, j] * 1000) / (rho * cp)
            
            T_new[i, j] = T_ext[i_ext, j] + dt * deriv
                
    return T_new


def transient_run_2D(time, Nr, Ntheta, dr, dtheta, dt, initial_condition, q, r, k, rho, cp):
    """
    Run transient 2D heat shield simulation.
    
    Parameters:
    -----------
    time : float
        Total simulation time
    Nr : int
        Number of radial nodes
    Ntheta : int
        Number of angular nodes
    dr : float
        Radial grid spacing
    dtheta : float
        Angular grid spacing
    dt : float
        Time step
    initial_condition : ndarray
        Initial temperature field
    q : ndarray
        Heat flux distribution
    r : ndarray
        Radial coordinates
    k : float
        Thermal conductivity
    rho : float
        Density
    cp : float
        Specific heat
    
    Returns:
    --------
    transient : list
        List of temperature fields at each time step
    """
    num_runs = int(time / dt)
    T_old = initial_condition 
    transient = []
    for i in range(num_runs):
        S_new = time_step_2D(Nr, Ntheta, dr, dtheta, dt, T_old, q, r, k, rho, cp)
        transient.append(S_new)
        T_old = S_new
        if i % 100 == 0:
            print(f"Time step {i}/{num_runs}")
    return transient


def time_step_3D(Nr, Ntheta, Nz, dr, dtheta, dz, dt, T, q, r, k, rho, cp):
    """
    Perform one time step for 3D heat shield (r-θ-z) using ghost nodes.
    
    Boundary Conditions:
    - Top surface (z=0): Neumann BC with heat flux q(r,θ)
    - Bottom surface (z=thickness): Insulated (zero flux)
    - Radial boundary (r=R): Insulated (zero flux)
    - Center (r=0): Handled with L'Hôpital's rule
    
    Parameters:
    -----------
    Nr : int
        Number of radial nodes
    Ntheta : int
        Number of angular nodes
    Nz : int
        Number of axial nodes
    dr : float
        Radial grid spacing
    dtheta : float
        Angular grid spacing
    dz : float
        Axial grid spacing
    dt : float
        Time step
    T : ndarray
        Current temperature field (Nr, Ntheta, Nz)
    q : ndarray
        Heat flux distribution (Nr, Ntheta) [kW/m²]
    r : ndarray
        Radial coordinates
    k : float
        Thermal conductivity
    rho : float
        Density
    cp : float
        Specific heat
    
    Returns:
    --------
    T_new : ndarray
        Updated temperature field
    """
    T_old = T.copy()
    alpha = k / (rho * cp)
    
    # Create extended array with ghost nodes: (Nr+1, Ntheta, Nz+2)
    # Interior: T_ext[1:Nr+1, :, 1:Nz+1] = T
    # Ghost nodes: T_ext[0, :, :] (r=0), T_ext[Nr, :, :] (r=R)
    #              T_ext[:, :, 0] (z=-dz), T_ext[:, :, Nz+1] (z=thickness+dz)
    T_ext = np.zeros((Nr+1, Ntheta, Nz+2))
    T_ext[1:Nr+1, :, 1:Nz+1] = T_old
    
    # Set ghost nodes based on boundary conditions
    # Top surface (z=0): Neumann BC -k * ∂T/∂z = q
    # Using centered difference: ∂T/∂z ≈ (T[1] - T[-1]) / (2*dz)
    # So: -k * (T[1] - T[-1]) / (2*dz) = q
    # Rearranging: -k*T[1] + k*T[-1] = 2*q*dz
    # Therefore: T[-1] = T[1] + (2*q*dz / k)
    # Note: q is in kW/m², convert to W/m²
    # Positive q means heat flowing INTO domain (heating), so T[-1] > T[1]
    for i in range(1, Nr+1):
        for j in range(Ntheta):
            # Top surface ghost node (z=-dz)
            T_ext[i, j, 0] = T_ext[i, j, 1] + (2 * q[i-1, j] * 1000 * dz / k)
            # Bottom surface ghost node (z=thickness+dz) - insulated
            T_ext[i, j, Nz+1] = T_ext[i, j, Nz]
    
    # Radial boundary ghost nodes (r=R) - insulated
    for j in range(Ntheta):
        for k_idx in range(1, Nz+1):
            T_ext[Nr, j, k_idx] = T_ext[Nr-1, j, k_idx]
    
    # Compute derivatives using ghost nodes
    T_new = np.zeros((Nr, Ntheta, Nz))
    
    for i in range(Nr):
        for j in range(Ntheta):
            for k_idx in range(Nz):
                # Map to extended array indices
                i_ext = i + 1
                k_ext = k_idx + 1
                
                # Handle r=0 singularity using L'Hôpital's rule
                if i == 0:
                    # At r=0: (1/r)*∂T/∂r → ∂²T/∂r² as r→0
                    # Modified radial term: 2 * (T[1] - T[0]) / dr²
                    radial_term = 2 * (T_ext[i_ext+1, j, k_ext] - T_ext[i_ext, j, k_ext]) / dr**2
                    # Angular term vanishes by symmetry at r=0
                    angular_term = 0
                elif i == Nr - 1:
                    # At radial boundary (r=R), use ghost node
                    # Standard radial derivatives using ghost node
                    radial_term = ((T_ext[Nr, j, k_ext] - 2*T_ext[i_ext, j, k_ext] + T_ext[i_ext-1, j, k_ext]) / dr**2 +
                                  (1/r[i])*(T_ext[Nr, j, k_ext] - T_ext[i_ext-1, j, k_ext]) / (2*dr))
                    
                    # Angular derivatives
                    if j == Ntheta - 1:
                        angular_term = (1/r[i]**2) * (T_ext[i_ext, 0, k_ext] - 2*T_ext[i_ext, j, k_ext] + T_ext[i_ext, j-1, k_ext]) / dtheta**2
                    else:
                        angular_term = (1/r[i]**2) * (T_ext[i_ext, j+1, k_ext] - 2*T_ext[i_ext, j, k_ext] + T_ext[i_ext, j-1, k_ext]) / dtheta**2
                else:
                    # Standard radial derivatives
                    radial_term = ((T_ext[i_ext+1, j, k_ext] - 2*T_ext[i_ext, j, k_ext] + T_ext[i_ext-1, j, k_ext]) / dr**2 +
                                  (1/r[i])*(T_ext[i_ext+1, j, k_ext] - T_ext[i_ext-1, j, k_ext]) / (2*dr))
                    
                    # Angular derivatives
                    if j == Ntheta - 1:
                        angular_term = (1/r[i]**2) * (T_ext[i_ext, 0, k_ext] - 2*T_ext[i_ext, j, k_ext] + T_ext[i_ext, j-1, k_ext]) / dtheta**2
                    else:
                        angular_term = (1/r[i]**2) * (T_ext[i_ext, j+1, k_ext] - 2*T_ext[i_ext, j, k_ext] + T_ext[i_ext, j-1, k_ext]) / dtheta**2
                
                # Axial derivatives using ghost nodes
                axial_term = (T_ext[i_ext, j, k_ext+1] - 2*T_ext[i_ext, j, k_ext] + T_ext[i_ext, j, k_ext-1]) / dz**2
                
                # Compute derivative
                deriv = alpha * (radial_term + angular_term + axial_term)
                
                # Add source term only at top surface (k_idx=0)
                if k_idx == 0:
                    # Source term is already incorporated via ghost node BC
                    pass
                
                T_new[i, j, k_idx] = T_ext[i_ext, j, k_ext] + dt * deriv
    
    return T_new


def transient_run_3D(time, Nr, Ntheta, Nz, dr, dtheta, dz, dt, initial_condition, 
                     q, r, k, rho, cp):
    """
    Run transient 3D heat shield simulation.
    
    Parameters:
    -----------
    time : float
        Total simulation time
    Nr : int
        Number of radial nodes
    Ntheta : int
        Number of angular nodes
    Nz : int
        Number of axial nodes
    dr : float
        Radial grid spacing
    dtheta : float
        Angular grid spacing
    dz : float
        Axial grid spacing
    dt : float
        Time step
    initial_condition : ndarray
        Initial temperature field
    q : ndarray
        Heat flux distribution
    r : ndarray
        Radial coordinates
    k : float
        Thermal conductivity
    rho : float
        Density
    cp : float
        Specific heat
    
    Returns:
    --------
    transient : list
        List of temperature fields at each time step
    """
    num_runs = int(time / dt)
    T_old = initial_condition 
    transient = []
    for i in range(num_runs):
        S_new = time_step_3D(Nr, Ntheta, Nz, dr, dtheta, dz, dt, T_old, q, r, k, rho, cp)
        transient.append(S_new)
        T_old = S_new
        if i % 100 == 0:
            print(f"Time step {i}/{num_runs}")
    return transient


def solve_heat_shield_2d(Nr, Ntheta, R, time, dt, k, rho, cp, initial_temp=0):
    """
    Solve 2D heat shield problem.
    
    Parameters:
    -----------
    Nr : int
        Number of radial nodes
    Ntheta : int
        Number of angular nodes
    R : float
        Radius
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
    initial_temp : float
        Initial temperature
    
    Returns:
    --------
    r, theta : ndarray
        Radial and angular coordinates
    transient : list
        List of temperature fields
    """
    dr = R / (Nr - 1)
    dtheta = 2*np.pi / (Ntheta - 1)
    
    # Create flux distribution
    r, theta, q = create_flux_distribution(Nr, Ntheta)
    
    # Initial condition
    initial_condition = np.zeros((Nr, Ntheta))
    
    # Run simulation
    transient = transient_run_2D(time, Nr, Ntheta, dr, dtheta, dt, initial_condition, 
                                 q, r, k, rho, cp)
    
    return r, theta, transient


def solve_heat_shield_3d(Nr, Ntheta, Nz, R, thickness, time, dt, k, rho, cp, initial_temp=0):
    """
    Solve 3D heat shield problem.
    
    Parameters:
    -----------
    Nr : int
        Number of radial nodes
    Ntheta : int
        Number of angular nodes
    Nz : int
        Number of axial nodes
    R : float
        Radius
    thickness : float
        Shield thickness
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
    initial_temp : float
        Initial temperature
    
    Returns:
    --------
    r, theta, z : ndarray
        Radial, angular, and axial coordinates
    transient : list
        List of temperature fields
    """
    dr = R / (Nr - 1)
    dtheta = 2*np.pi / (Ntheta - 1)
    dz = thickness / Nz
    
    # Create flux distribution
    r, theta, q = create_flux_distribution(Nr, Ntheta)
    z = np.linspace(0, thickness, Nz)
    
    # Initial condition
    initial_condition = np.zeros((Nr, Ntheta, Nz))
    
    # Run simulation
    transient = transient_run_3D(time, Nr, Ntheta, Nz, dr, dtheta, dz, dt, 
                                 initial_condition, q, r, k, rho, cp)
    
    return r, theta, z, transient

