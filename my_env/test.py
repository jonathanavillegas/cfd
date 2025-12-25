
import numpy as np
from scipy.interpolate import interp1d

def create_flux_distribution(Nr, Ntheta):
    # Digitized radial (Y-axis in mm) and heat flux (kW/m^2) values from graph
    r_data = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
                   33, 36, 39, 42, 45, 48, 51, 54, 57,
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75])
    q_data = np.array([87.5, 86, 84, 80, 75, 69, 64, 60, 56, 53, 51,
                   48, 46, 44, 42, 41, 40, 39, 38.5, 38,
                   38, 40, 43, 48, 54, 62, 66, 70, 74, 77,
                   76, 74, 71, 68, 64, 60])
    
    # Interpolation function
    flux_interp = interp1d(r_data, q_data, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    # Define radial and angular positions
    r = np.linspace(0, r_data[-1], n_r)
    theta = np.linspace(0, 2*np.pi, Ntheta)
    
    # Generate 2D flux array: same value across each theta for a given r
    flux_2D = np.zeros((Nr, Ntheta))
    for i, r in enumerate(r_vals):
        flux_2D[i, :] = flux_interp(r)
    
    return r, theta, flux_2D

import matplotlib.pyplot as plt

n_r, n_theta = 100, 180
r_vals, theta_vals, flux = create_flux_distribution(n_r, n_theta)

# Convert to Cartesian coordinates for plotting
R, Theta = np.meshgrid(r_vals, theta_vals, indexing='ij')
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

plt.figure(figsize=(6, 6))
plt.pcolormesh(X, Y, flux, shading='auto', cmap='hot')
plt.colorbar(label='Heat flux (kW/m²)')
plt.title('Heat Flux Distribution on Circular Disk')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.axis('equal')
plt.show()
