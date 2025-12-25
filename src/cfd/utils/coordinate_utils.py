"""
Coordinate transformation utilities for cylindrical to Cartesian conversion.
"""
import numpy as np


def cylindrical_to_cartesian(r, theta, z=None):
    """
    Convert cylindrical coordinates to Cartesian.
    
    Parameters:
    -----------
    r : ndarray
        Radial coordinates
    theta : ndarray
        Angular coordinates
    z : ndarray, optional
        Axial coordinates (if None, returns 2D)
    
    Returns:
    --------
    X, Y : ndarray
        Cartesian x and y coordinates
    Z : ndarray, optional
        Cartesian z coordinates (if z provided)
    """
    if z is None:
        # 2D case
        R_grid, Theta_grid = np.meshgrid(r, theta, indexing='ij')
        X = R_grid * np.cos(Theta_grid)
        Y = R_grid * np.sin(Theta_grid)
        return X, Y
    else:
        # 3D case
        R_grid, Theta_grid, Z_grid = np.meshgrid(r, theta, z, indexing='ij')
        X = R_grid * np.cos(Theta_grid)
        Y = R_grid * np.sin(Theta_grid)
        return X, Y, Z_grid

