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
        if i % (num_runs / 100) == 0:
            print(f"Time step {i}/{num_runs}")
    return transient


def time_step_3D(Nr, Ntheta, Nz, dr, dtheta, dz, dt, T, latent_heat, surface_bool, destroyed_bool, 
    q, r, k, rho, cp, sub_temp, sub_heat):
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
    sub_temp : float
        Sublimation temperature
    
    Returns:
    --------
    T_new : ndarray
        Updated temperature field
    """
    T_old = T.copy()
    latent_heat_old = latent_heat.copy()
    surface_bool_old = surface_bool.copy()
    destroyed_bool_old = destroyed_bool.copy()
    alpha = k / (rho * cp)
    
    # Create extended array with ghost nodes: (Nr+1, Ntheta, Nz+2)
    # Interior: T_ext[1:Nr+1, :, 1:Nz+1] = T
    # Ghost nodes: T_ext[0, :, :] (r=0), T_ext[Nr, :, :] (r=R)
    #              T_ext[:, :, 0] (z=-dz), T_ext[:, :, Nz+1] (z=thickness+dz)
    T_ext = np.zeros((Nr+1, Ntheta, Nz+2))
    T_ext[1:Nr+1, :, 1:Nz+1] = T_old
    
    #3 Classes of nodes:
    # 1. Surface nodes: nodes that are exposed to the heat flux
    # 2. Adjacent to destroyed nodes: nodes that are adjacent to a destroyed node
    # 3. Interior nodes: nodes that are not on the surface or adjacent to destroyed nodes
    # Using symmetry we can assume no nodes are adjacent to destroyed nodes in the azimuthal direction
    # Loop over extended array indices: i_ext = 1..Nr (corresponds to original i = 0..Nr-1)
    for i_ext in range(1, Nr+1):
        i_orig = i_ext - 1  # Original array index
        for j in range(Ntheta):
            for k_idx in range(0, Nz):
                k_ext = k_idx + 1  # Extended array k index
                # If surface node, set the node above to be ghost node
                if destroyed_bool_old[i_orig, j, k_idx] == False:
                    if surface_bool_old[i_orig, j, k_idx] == True:
                        # Surface node at k_idx=0: ghost node is at T_ext[i_ext, j, 0], surface is at T_ext[i_ext, j, 1]
                        # Ghost node BC: T_ext[i_ext, j, 0] = T_ext[i_ext, j, 1] + (2 * q * dz / k)
                        T_ext[i_ext, j, k_ext-1] = T_ext[i_ext, j, k_ext + 1] + (2 * q[i_orig, j] * 1000 * dz / k)
                    # If adjacent in radial direction to destroyed node set to ghost node
                    elif i_orig > 0 and i_orig < Nr - 1:
                        # Check left neighbor (i_orig - 1)
                        if destroyed_bool_old[i_orig - 1, j, k_idx] == True:
                            T_ext[i_ext - 1, j, k_ext] = T_ext[i_ext, j, k_ext] + q[i_orig - 1, j] * 1000 * dr / k
                        # Check right neighbor (i_orig + 1)
                        if destroyed_bool_old[i_orig + 1, j, k_idx] == True:
                            T_ext[i_ext + 1, j, k_ext] = T_ext[i_ext, j, k_ext] + q[i_orig + 1, j] * 1000 * dr / k
                    
            # Bottom surface ghost node (z=thickness+dz) - insulated
            T_ext[i_ext, j, Nz+1] = T_ext[i_ext, j, Nz]
            
    
    # Radial boundary ghost nodes (r=R) - insulated
    for j in range(Ntheta):
        for k_idx in range(1, Nz+1):
            T_ext[Nr, j, k_idx] = T_ext[Nr-1, j, k_idx]
    
    # Compute derivatives using ghost nodes
    T_new = np.zeros((Nr, Ntheta, Nz))
    latent_heat_new = latent_heat_old.copy()
    surface_bool_new = surface_bool_old.copy()
    destroyed_bool_new = destroyed_bool_old.copy()
    for i in range(Nr):
        for j in range(Ntheta):
            for k_idx in range(Nz):
                # Map to extended array indices
                i_ext = i + 1
                k_ext = k_idx + 1
                
                if destroyed_bool_old[i, j, k_idx] == False:
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

                    # Update latent heat, surface boolean, and destroyed boolean arrays
                    if T_new[i, j, k_idx] >= sub_temp:
                        latent_heat_new[i, j, k_idx] = latent_heat_old[i, j, k_idx] + (T_new[i, j, k_idx] - sub_temp) * rho * cp
                        T_new[i, j, k_idx] = sub_temp
                    if latent_heat_new[i, j, k_idx] >= sub_heat:
                        destroyed_bool_new[i, j, k_idx] = True
                        if surface_bool_old[i, j, k_idx] == True:
                            surface_bool_new[i, j, k_idx] = False
                            surface_bool_new[i, j, k_idx + 1] = True
    

    return T_new, latent_heat_new, surface_bool_new, destroyed_bool_new

def transient_run_3D(time, Nr, Ntheta, Nz, dr, dtheta, dz, dt, initial_condition, 
                     q, r, k, rho, cp, latent_heat, surface_bool, destroyed_bool, sub_temp, sub_heat):
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
    sub_temp : float
        Sublimation temperature
    sub_heat : float
        Sublimation heat
    
    Returns:
    --------
    transient : list
        List of temperature fields at each time step
    """
    num_runs = int(time / dt)
    T_old = initial_condition 
    latent_heat_old = latent_heat
    surface_bool_old = surface_bool
    destroyed_bool_old = destroyed_bool
    transient = []
    for i in range(num_runs):
        S_new, latent_heat_new, surface_bool_new, destroyed_bool_new = time_step_3D(Nr, Ntheta, Nz, dr, dtheta, dz, dt, T_old, latent_heat_old, surface_bool_old, destroyed_bool_old, q, r, k, rho, cp, sub_temp, sub_heat)
        transient.append(S_new)
        T_old = S_new
        latent_heat_old = latent_heat_new
        surface_bool_old = surface_bool_new
        destroyed_bool_old = destroyed_bool_new
        if i % 10 == 0:
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

def create_data_arrays(Nr, Ntheta, Nz):
    """
    Create latent heat, surface boolean, and destroyed boolean arrays to track phase change and surface location
    Latent heat: amount of heat required to change from solid to gas
    Surface boolean: whether the surface is exposed to the heat flux
    Destroyed boolean: whether the surface is destroyed
    """
    latent_heat = np.zeros((Nr, Ntheta, Nz)) # in J/kg
    surface_bool = np.zeros((Nr, Ntheta, Nz), dtype=bool)
    surface_bool[:, :, 0] = True
    destroyed_bool = np.zeros((Nr, Ntheta, Nz), dtype=bool)
    return latent_heat, surface_bool, destroyed_bool

def solve_heat_shield_3d(Nr, Ntheta, Nz, R, thickness, time, dt, k, rho, cp, initial_temp, sub_temp, sub_heat):
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

    # Create latent heat and surface boolean arrays to track phase change and surface location
    latent_heat, surface_bool, destroyed_bool = create_data_arrays(Nr, Ntheta, Nz)

    # Run simulation
    transient = transient_run_3D(time, Nr, Ntheta, Nz, dr, dtheta, dz, dt, 
                                 initial_condition, q, r, k, rho, cp, latent_heat, surface_bool, destroyed_bool, sub_temp, sub_heat)
    
    return r, theta, z, transient

