"""
3D visualization functions for heat transfer simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .plot_2d import plot_cylindrical_2d_animated


def plot_rz_cross_sections(T, r, theta, z, time, dt, theta_idx=0, n_snapshots=None):
    """
    Plot r-z cross-sections at multiple time steps with adaptive intervals.
    
    Parameters:
    -----------
    T : ndarray or list
        Temperature fields at different times (Nt, Nr, Ntheta, Nz)
    r : ndarray
        Radial coordinates
    theta : ndarray
        Angular coordinates
    z : ndarray
        Axial coordinates
    time : float
        Total simulation time
    dt : float
        Time step
    theta_idx : int
        Angular index for cross-section (default 0, i.e., theta=0)
    n_snapshots : int, optional
        Number of snapshots to show. If None, automatically determined based on time.
    """
    if isinstance(T, list):
        T = np.array(T)
    
    # If single time step, add dimension
    if T.ndim == 3:
        T = T[np.newaxis, :, :, :]
    
    n_times = len(T)
    
    # Determine number of snapshots adaptively
    if n_snapshots is None:
        # Show more snapshots for longer simulations
        if time <= 10:
            n_snapshots = 5
        elif time <= 50:
            n_snapshots = 8
        elif time <= 100:
            n_snapshots = 12
        else:
            n_snapshots = 15
    
    # Calculate time indices evenly spaced
    if n_snapshots >= n_times:
        time_indices = list(range(n_times))
    else:
        time_indices = [int(i * (n_times - 1) / (n_snapshots - 1)) for i in range(n_snapshots)]
    
    # Create subplot grid
    n_cols = 4
    n_rows = (len(time_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    
    # Ensure axes is always 2D array
    if n_rows == 1:
        if len(time_indices) == 1:
            axes = np.array([[axes]])
        else:
            axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get common color scale
    vmin = np.min(T)
    vmax = np.max(T)
    
    for idx, t_idx in enumerate(time_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Extract r-z cross-section at theta_idx
        T_slice = T[t_idx]
        rz_slice = T_slice[:, theta_idx, :]
        
        # Create meshgrid for plotting
        r_vals = np.linspace(0, r[-1], len(r))
        z_vals = np.linspace(0, z[-1], len(z))
        R_grid, Z_grid = np.meshgrid(r_vals, z_vals, indexing='ij')
        
        # Plot
        im = ax.pcolormesh(Z_grid, R_grid, rz_slice, cmap='hot', shading='auto', 
                          vmin=vmin, vmax=vmax)
        ax.set_xlabel('z [m]')
        ax.set_ylabel('r [m]')
        ax.set_title(f't = {t_idx*dt:.2f} s')
        plt.colorbar(im, ax=ax, label='Temperature [K]')
    
    # Hide unused subplots
    for idx in range(len(time_indices), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'r-z Cross Section (θ={theta[theta_idx]:.2f} rad) - Temperature Evolution', 
                 fontsize=14, y=1.0)
    plt.tight_layout()
    plt.show()


def plot_surface_temperature_evolution(T, r, theta, dt, z_surface_idx=0, title='Surface Temperature Evolution'):
    """
    Plot animated surface temperature evolution for 3D heat shield.
    Shows the r-θ plane at the surface (z=0) over time, similar to 2D solver visualization.
    
    Parameters:
    -----------
    T : ndarray or list
        Temperature fields at different times (Nt, Nr, Ntheta, Nz)
    r : ndarray
        Radial coordinates
    theta : ndarray
        Angular coordinates
    dt : float
        Time step
    z_surface_idx : int
        Index of surface layer (default 0 for z=0)
    title : str
        Plot title
    """
    if isinstance(T, list):
        T = np.array(T)
    
    # Extract surface temperature (z=0 layer)
    # T has shape (Nt, Nr, Ntheta, Nz), extract surface at z_surface_idx
    surface_T = T[:, :, :, z_surface_idx]  # Shape: (Nt, Nr, Ntheta)
    
    # Use the existing cylindrical 2D animated plot
    plot_cylindrical_2d_animated(surface_T, r, theta, dt, title=title)


def plot_rz_cross_section_animated(T, r, theta, z, dt, theta_idx=0, title='r-z Cross Section Temperature Evolution'):
    """
    Plot animated r-z cross-section showing temperature evolution over time.
    
    Parameters:
    -----------
    T : ndarray or list
        Temperature fields at different times (Nt, Nr, Ntheta, Nz)
    r : ndarray
        Radial coordinates
    theta : ndarray
        Angular coordinates
    z : ndarray
        Axial coordinates
    dt : float
        Time step
    theta_idx : int
        Angular index for cross-section (default 0, i.e., theta=0)
    title : str
        Plot title
    """
    if isinstance(T, list):
        T = np.array(T)
    
    # If single time step, add dimension
    if T.ndim == 3:
        T = T[np.newaxis, :, :, :]
    
    # Extract r-z cross-section at theta_idx for all time steps
    # T has shape (Nt, Nr, Ntheta, Nz)
    rz_frames = T[:, :, theta_idx, :]  # Shape: (Nt, Nr, Nz)
    
    # Create meshgrid for plotting (matching plot_rz_cross_sections)
    r_vals = np.linspace(0, r[-1], len(r))
    z_vals = np.linspace(0, z[-1], len(z))
    R_grid, Z_grid = np.meshgrid(r_vals, z_vals, indexing='ij')
    
    # Get common color scale
    vmin = np.min(T)
    vmax = np.max(T)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initial plot (R_grid is x-axis, Z_grid is y-axis - switched from original)
    # rz_frames[0] has shape (Nr, Nz) which matches R_grid and Z_grid shape
    im = ax.pcolormesh(R_grid, Z_grid, rz_frames[0], cmap='hot', shading='auto', 
                      vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, label='Temperature [K]')
    ax.set_xlabel('r [m]')
    ax.set_ylabel('z [m]')
    ax.invert_yaxis()  # Invert y-axis so z=0 (surface) is at the top
    ax.set_title(f'{title} - Time = 0.0 s\n(θ={theta[theta_idx]:.2f} rad)')
    
    def update(frame):
        im.set_array(rz_frames[frame].flatten())
        ax.set_title(f'{title} - Time = {frame * dt:.2f} s\n(θ={theta[theta_idx]:.2f} rad)')
        return [im]
    
    ani = animation.FuncAnimation(
        fig, update, frames=rz_frames.shape[0], interval=100, blit=False
    )
    plt.tight_layout()
    plt.show()
