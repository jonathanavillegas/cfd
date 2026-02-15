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
    q, r, k, rho, cp, sub_temp, sub_heat, density, rho_residual, k_pyro_A, k_pyro_Ea, delta_H_p):
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
        Initial/reference density
    cp : float
        Specific heat
    sub_temp : float
        Sublimation temperature
    density : ndarray
        Current density field (Nr, Ntheta, Nz) [kg/m³]
    rho_residual : float
        Residual density (char/gas density) [kg/m³]
    k_pyro_A : float
        Pyrolysis rate pre-exponential factor [1/s]
    k_pyro_Ea : float
        Pyrolysis activation energy [J/mol]
    delta_H_p : float
        Enthalpy of pyrolysis (ΔH_p) [J/kg]
    
    Returns:
    --------
    T_new : ndarray
        Updated temperature field
    latent_heat_new : ndarray
        Updated latent heat field
    surface_bool_new : ndarray
        Updated surface boolean field
    destroyed_bool_new : ndarray
        Updated destroyed boolean field
    density_new : ndarray
        Updated density field
    """
    T_old = T.copy()
    latent_heat_old = latent_heat.copy()
    surface_bool_old = surface_bool.copy()
    destroyed_bool_old = destroyed_bool.copy()
    density_old = density.copy()
    
    # Use average density for thermal diffusivity calculation
    rho_avg = np.mean(density_old[density_old > rho_residual]) if np.any(density_old > rho_residual) else rho
    alpha = k / (rho_avg * cp)
    
    R_gas = 8.314  # Universal gas constant [J/(mol·K)]
    
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
                        # Surface node: ghost node BC for Neumann condition -k * ∂T/∂z = q
                        # Using centered difference: T_ghost = T_surface + (2 * q * dz / k)
                        # q is in kW/m², convert to W/m² (multiply by 1000)
                        # Note: q is already per unit area, so no need to multiply by dr*dtheta
                        T_ext[i_ext, j, k_ext-1] = T_ext[i_ext, j, k_ext + 1] + (2 * q[i_orig, j] * 1000 * dz / k)
                    # If adjacent in radial direction to destroyed node set to ghost node
                    elif i_orig > 0 and i_orig < Nr - 1:
                        # Check left neighbor (i_orig - 1)
                        if destroyed_bool_old[i_orig - 1, j, k_idx] == True:
                            T_ext[i_ext - 1, j, k_ext] = 0
                            #T_ext[i_ext - 1, j, k_ext] = T_ext[i_ext, j, k_ext] + (q[i_orig - 1, j] * 1000 * dr / k)
                        # Check right neighbor (i_orig + 1)
                        if destroyed_bool_old[i_orig + 1, j, k_idx] == True:
                            T_ext[i_ext + 1, j, k_ext] = 0
                            #T_ext[i_ext + 1, j, k_ext] = T_ext[i_ext, j, k_ext] + (q[i_orig + 1, j] * 1000 * dr / k)
                    
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
    density_new = density_old.copy()  # Track density evolution
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
                    
                    # Axial derivatives
                    axial_term = (T_ext[i_ext, j, k_ext+1] - 2*T_ext[i_ext, j, k_ext] + T_ext[i_ext, j, k_ext-1]) / dz**2
                    
                    # Compute derivative (conduction terms)
                    deriv = alpha * (radial_term + angular_term + axial_term)
                    
                    # Pyrolysis term: ΔH_p (dρ/dt) / (ρ * cp)
                    # Density evolution: dρ/dt = -k(T)(ρ - ρ_r)
                    pyrolysis_term = 0.0
                    if density_old[i, j, k_idx] > rho_residual and T_ext[i_ext, j, k_ext] > 0:
                        # Pyrolysis rate constant (Arrhenius form)
                        # k(T) = A * exp(-Ea / (R*T))
                        k_pyro = k_pyro_A * np.exp(-k_pyro_Ea / (R_gas * T_ext[i_ext, j, k_ext]))
                        
                        # Density rate: dρ/dt = -k(T)(ρ - ρ_r)
                        density_rate = -k_pyro * (density_old[i, j, k_idx] - rho_residual)
                        
                        # Update density
                        density_new[i, j, k_idx] = density_old[i, j, k_idx] + density_rate * dt
                        density_new[i, j, k_idx] = max(density_new[i, j, k_idx], rho_residual)  # Clamp to residual
                        
                        # Pyrolysis term: ΔH_p (dρ/dt) / (ρ * cp)
                        # Note: density_rate is negative (density decreasing), so this is an energy sink
                        pyrolysis_term = (delta_H_p * density_rate) / (density_old[i, j, k_idx] * cp)
                        
                    
                    # Add source term only at top surface (k_idx=0)
                    if k_idx == 0:
                        # Source term is already incorporated via ghost node BC
                        pass
                    
                    T_new[i, j, k_idx] = T_ext[i_ext, j, k_ext] + dt * (deriv + pyrolysis_term)

                    # Update latent heat, surface boolean, and destroyed boolean arrays
                    if T_new[i, j, k_idx] >= sub_temp:
                        latent_heat_new[i, j, k_idx] = latent_heat_old[i, j, k_idx] + (T_new[i, j, k_idx] - sub_temp) * rho * cp
                        T_new[i, j, k_idx] = sub_temp
                    if latent_heat_new[i, j, k_idx] >= sub_heat:
                        destroyed_bool_new[i, j, k_idx] = True
                        if surface_bool_old[i, j, k_idx] == True:
                            surface_bool_new[i, j, k_idx] = False
                            if k_idx + 1 < Nz:
                                surface_bool_new[i, j, k_idx + 1] = True
                    
                    

    return T_new, latent_heat_new, surface_bool_new, destroyed_bool_new, density_new

def transient_run_3D(time, Nr, Ntheta, Nz, dr, dtheta, dz, dt, initial_condition, 
                     q, r, k, rho, cp, latent_heat, surface_bool, destroyed_bool, 
                     sub_temp, sub_heat, density, rho_residual, k_pyro_A, k_pyro_Ea, delta_H_p):
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
        Initial/reference density
    cp : float
        Specific heat
    sub_temp : float
        Sublimation temperature
    sub_heat : float
        Sublimation heat
    density : ndarray
        Initial density field
    rho_residual : float
        Residual density (char/gas) [kg/m³]
    k_pyro_A : float
        Pyrolysis pre-exponential [1/s]
    k_pyro_Ea : float
        Pyrolysis activation energy [J/mol]
    delta_H_p : float
        Pyrolysis enthalpy [J/kg]
    
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
    density_old = density
    transient = []
    for i in range(num_runs):
        T_new, latent_heat_new, surface_bool_new, destroyed_bool_new, density_new = time_step_3D(
            Nr, Ntheta, Nz, dr, dtheta, dz, dt, T_old, latent_heat_old, surface_bool_old, 
            destroyed_bool_old, q, r, k, rho, cp, sub_temp, sub_heat, density_old, 
            rho_residual, k_pyro_A, k_pyro_Ea, delta_H_p
        )
        transient.append(T_new)
        T_old = T_new
        latent_heat_old = latent_heat_new
        surface_bool_old = surface_bool_new
        destroyed_bool_old = destroyed_bool_new
        density_old = density_new  # Update density for next iteration
        if i % 10 == 0:
            print(f"Time step {i}/{num_runs}")
    return transient


def solve_heat_shield_2d(Nr, Ntheta, R, time, dt, k, rho, cp, initial_temp=0, distribution_type=1):
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
    
    # Generate r coordinates based on actual domain radius R
    r = np.linspace(0, R, Nr)
    
    # Create flux distribution evaluated at actual grid coordinates
    r, theta, q = create_flux_distribution(Nr, Ntheta, distribution_type=distribution_type, r_coords=r)
    
    # Initial condition
    initial_condition = np.zeros((Nr, Ntheta))
    
    # Run simulation
    transient = transient_run_2D(time, Nr, Ntheta, dr, dtheta, dt, initial_condition, 
                                 q, r, k, rho, cp)
    
    return r, theta, transient

def create_data_arrays(Nr, Ntheta, Nz, rho_initial, rho_residual):
    """
    Create latent heat, surface boolean, destroyed boolean, and density arrays to track phase change and surface location.
    Latent heat: amount of heat required to change from solid to gas
    Surface boolean: whether the surface is exposed to the heat flux
    Destroyed boolean: whether the surface is destroyed
    Density: current material density (decreases during pyrolysis)
    
    Parameters:
    -----------
    rho_initial : float
        Initial solid density [kg/m³]
    rho_residual : float
        Residual density (char/gas) [kg/m³]
    """
    latent_heat = np.zeros((Nr, Ntheta, Nz))  # in J/kg
    surface_bool = np.zeros((Nr, Ntheta, Nz), dtype=bool)
    surface_bool[:, :, 0] = True
    destroyed_bool = np.zeros((Nr, Ntheta, Nz), dtype=bool)
    density = np.ones((Nr, Ntheta, Nz)) * rho_initial  # Initialize to solid density
    return latent_heat, surface_bool, destroyed_bool, density

def solve_heat_shield_3d(Nr, Ntheta, Nz, R, thickness, time, dt, k, rho, cp, initial_temp, sub_temp, sub_heat,
                         rho_residual, k_pyro_A, k_pyro_Ea, delta_H_p, distribution_type):
    """
    Solve 3D heat shield problem with pyrolysis and sublimation.
    
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
        Initial density
    cp : float
        Specific heat
    initial_temp : float
        Initial temperature
    sub_temp : float
        Sublimation temperature
    sub_heat : float
        Latent heat of sublimation
    distribution_type : int
        Heat flux distribution type (1 or 2)
    rho_residual : float
        Residual density (char/gas) [kg/m³], default 50
    k_pyro_A : float
        Pyrolysis pre-exponential [1/s], default 1e6
    k_pyro_Ea : float
        Pyrolysis activation energy [J/mol], default 100000
    delta_H_p : float
        Pyrolysis enthalpy [J/kg], default 2.5e6
    
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
    
    # Generate r coordinates based on actual domain radius R
    r = np.linspace(0, R, Nr)
    
    # Create flux distribution evaluated at actual grid coordinates
    r, theta, q = create_flux_distribution(Nr, Ntheta, distribution_type, r_coords=r)
    z = np.linspace(0, thickness, Nz)
    
    # Initial condition
    initial_condition = np.zeros((Nr, Ntheta, Nz))

    # Create data arrays including density
    latent_heat, surface_bool, destroyed_bool, density = create_data_arrays(
        Nr, Ntheta, Nz, rho, rho_residual
    )

    # Run simulation
    transient = transient_run_3D(time, Nr, Ntheta, Nz, dr, dtheta, dz, dt, 
                                 initial_condition, q, r, k, rho, cp, 
                                 latent_heat, surface_bool, destroyed_bool,
                                 sub_temp, sub_heat, density, rho_residual,
                                 k_pyro_A, k_pyro_Ea, delta_H_p)
    
    return r, theta, z, transient

