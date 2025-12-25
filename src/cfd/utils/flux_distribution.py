"""
Heat flux distribution functions for heat shield simulations.
"""
import numpy as np
from scipy.interpolate import interp1d


def create_flux_distribution(Nr, Ntheta):
    """
    Create heat flux distribution for cylindrical heat shield.
    
    Parameters:
    -----------
    Nr : int
        Number of radial nodes
    Ntheta : int
        Number of angular nodes
    
    Returns:
    --------
    r : ndarray
        Radial coordinates
    theta : ndarray
        Angular coordinates
    flux : ndarray
        Heat flux array of shape (Nr, Ntheta) [kW/m²]
    """
    r_data = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
                   33, 36, 39, 42, 45, 48, 51, 54, 57,
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75])
    q_data = np.array([87.5, 86, 84, 80, 75, 69, 64, 60, 56, 53, 51,
                   48, 46, 44, 42, 41, 40, 39, 38.5, 38,
                   38, 40, 43, 48, 54, 62, 66, 70, 74, 77,
                   76, 74, 71, 68, 64, 60])
     
    flux_interp = interp1d(r_data, q_data, kind='cubic', bounds_error=False, fill_value="extrapolate")
    r = np.linspace(.000001, r_data[-1], Nr)
    theta = np.linspace(.000001, 2*np.pi, Ntheta)
    
    flux = np.zeros((Nr, Ntheta))
    for i, j in enumerate(r):
        flux[i, :] = flux_interp(j)
    
    return r, theta, flux

