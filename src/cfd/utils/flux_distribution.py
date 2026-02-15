"""
Heat flux distribution functions for heat shield simulations.
"""
import numpy as np
from scipy.interpolate import interp1d


def create_flux_distribution(Nr, Ntheta, distribution_type=1, r_coords=None):
    """
    Create heat flux distribution for cylindrical heat shield.
    
    Parameters:
    -----------
    Nr : int
        Number of radial nodes
    Ntheta : int
        Number of angular nodes
    distribution_type : int
        Type of distribution to use:
        1 = Original distribution (r up to 75 m)
        2 = Vehicle shape distribution (r up to 0.4 m / 400 mm)
    r_coords : ndarray, optional
        Radial coordinates at which to evaluate flux. If None, uses default based on distribution_type.
    
    Returns:
    --------
    r : ndarray
        Radial coordinates [m] (same as r_coords if provided, otherwise default)
    theta : ndarray
        Angular coordinates [rad]
    flux : ndarray
        Heat flux array of shape (Nr, Ntheta) [kW/m²]
    """
    if distribution_type == 1:
        # Original distribution
        r_data = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
                       33, 36, 39, 42, 45, 48, 51, 54, 57,
                       60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                       70, 71, 72, 73, 74, 75])
        q_data = np.array([87.5, 86, 84, 80, 75, 69, 64, 60, 56, 53, 51,
                       48, 46, 44, 42, 41, 40, 39, 38.5, 38,
                       38, 40, 43, 48, 54, 62, 66, 70, 74, 77,
                       76, 74, 71, 68, 64, 60])
        r_max = 75.0  # [m]
    elif distribution_type == 2:
        # Vehicle shape distribution (from image: 0-400 mm, heat flux 28-44.5 kW/m²)
        # Convert mm to m for r_data
        r_data_mm = np.array([0, 50, 100, 150, 200, 250, 300, 315, 325, 330, 335, 340, 345, 350, 400])
        r_data = r_data_mm / 1000.0  # Convert mm to m
        # Heat flux data based on image description
        # Starts at ~34, decreases to ~32, rises to peak ~44.5 at 335mm, drops to ~28
        q_data = np.array([34.0, 33.5, 33.0, 32.0, 32.0, 33.0, 38.0, 42.0, 44.0, 44.3, 44.5, 42.0, 29.0, 28.0, 28.0])
        r_max = 0.4  # [m] = 400 mm
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}. Use 1 or 2.")
     
    # Use extrapolation for values outside bounds
    flux_interp = interp1d(r_data, q_data, kind='cubic', bounds_error=False, fill_value='extrapolate')  # type: ignore
    
    # Use provided r_coords if available, otherwise generate default
    if r_coords is not None:
        r = r_coords
    else:
        r = np.linspace(.000001, r_max, Nr)
    
    theta = np.linspace(.000001, 2*np.pi, Ntheta)
    
    flux = np.zeros((Nr, Ntheta))
    for i, j in enumerate(r):
        flux[i, :] = flux_interp(j)
    
    return r, theta, flux

