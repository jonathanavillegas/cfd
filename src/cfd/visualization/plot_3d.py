"""
3D visualization functions for heat transfer simulations.

Note: Full implementation with all functions will be added in subsequent tasks.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from ..utils.coordinate_utils import cylindrical_to_cartesian


def plot_3d_surface_animated(T_frames, r, theta, dt, title='Temperature Evolution'):
    """
    Animated 3D surface plot of top surface (z=0).
    
    Parameters:
    -----------
    T_frames : list or ndarray
        List of temperature fields at each time step (Nt, Nr, Ntheta)
    r : ndarray
        Radial coordinates
    theta : ndarray
        Angular coordinates
    dt : float
        Time step
    title : str
        Plot title
    """
    T_frames = np.array(T_frames)
    X, Y = cylindrical_to_cartesian(r, theta)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial surface
    surf = ax.plot_surface(X, Y, T_frames[0], cmap='hot', 
                           vmin=np.min(T_frames), vmax=np.max(T_frames))
    ax.set_title(f'{title} - Time = 0.0 s')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Temperature')
    fig.colorbar(surf, ax=ax, label='Temperature')
    
    def update(frame):
        ax.clear()
        surf = ax.plot_surface(X, Y, T_frames[frame], cmap='hot',
                              vmin=np.min(T_frames), vmax=np.max(T_frames))
        ax.set_title(f'{title} - Time = {frame * dt:.2f} s')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Temperature')
        return ax,
    
    ani = animation.FuncAnimation(fig, update, frames=len(T_frames), interval=100)
    plt.show()


def plot_3d_isosurface(T, r, theta, z, levels, title='Temperature Isosurfaces'):
    """
    3D isosurface plots showing temperature contour levels.
    
    Parameters:
    -----------
    T : ndarray
        Temperature field (Nr, Ntheta, Nz)
    r : ndarray
        Radial coordinates
    theta : ndarray
        Angular coordinates
    z : ndarray
        Axial coordinates
    levels : list
        Temperature levels to plot
    title : str
        Plot title
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Convert to Cartesian
        R_grid, Theta_grid, Z_grid = np.meshgrid(r, theta, z, indexing='ij')
        X = R_grid * np.cos(Theta_grid)
        Y = R_grid * np.sin(Theta_grid)
        
        fig = go.Figure()
        
        for level in levels:
            # Create isosurface
            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z_grid.flatten(),
                value=T.flatten(),
                isomin=level,
                isomax=level,
                surface_count=1,
                colorscale='hot',
                showscale=True,
                name=f'T = {level}K'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X [m]',
                yaxis_title='Y [m]',
                zaxis_title='Z [m]'
            )
        )
        fig.show()
    except ImportError:
        # Fallback to matplotlib
        print("Plotly not available, using matplotlib 3D scatter")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample points for each level
        for level in levels:
            # Find points near this temperature level
            mask = np.abs(T - level) < (np.max(T) - np.min(T)) * 0.05
            if np.any(mask):
                R_grid, Theta_grid, Z_grid = np.meshgrid(r, theta, z, indexing='ij')
                X = R_grid * np.cos(Theta_grid)
                Y = R_grid * np.sin(Theta_grid)
                ax.scatter(X[mask], Y[mask], Z_grid[mask], 
                          label=f'T = {level}K', alpha=0.3)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(title)
        ax.legend()
        plt.show()


def plot_cross_sections(T, r, theta, z, time_indices, dt):
    """
    Multiple cross-sectional views at different time snapshots.
    
    Parameters:
    -----------
    T : ndarray or list
        Temperature fields at different times (Nt, Nr, Ntheta, Nz) or single time (Nr, Ntheta, Nz)
    r : ndarray
        Radial coordinates
    theta : ndarray
        Angular coordinates
    z : ndarray
        Axial coordinates
    time_indices : list
        Time indices to plot
    dt : float
        Time step
    """
    if isinstance(T, list):
        T = np.array(T)
    
    # If single time step, add dimension
    if T.ndim == 3:
        T = T[np.newaxis, :, :, :]
    
    n_times = len(time_indices)
    fig, axes = plt.subplots(n_times, 3, figsize=(15, 5*n_times))
    
    if n_times == 1:
        axes = axes[np.newaxis, :]
    
    for idx, t_idx in enumerate(time_indices):
        T_slice = T[t_idx]
        
        # r-z cross-section (θ=0)
        theta_idx = 0
        rz_slice = T_slice[:, theta_idx, :]
        r_vals = np.linspace(0, r[-1], len(r))
        z_vals = np.linspace(0, z[-1], len(z))
        R_grid, Z_grid = np.meshgrid(r_vals, z_vals, indexing='ij')
        
        im1 = axes[idx, 0].pcolormesh(Z_grid, R_grid, rz_slice, cmap='hot', shading='auto')
        axes[idx, 0].set_xlabel('z [m]')
        axes[idx, 0].set_ylabel('r [m]')
        axes[idx, 0].set_title(f'r-z cross-section (θ=0), t={t_idx*dt:.2f}s')
        plt.colorbar(im1, ax=axes[idx, 0])
        
        # r-θ cross-section (z=midpoint)
        z_idx = len(z) // 2
        rtheta_slice = T_slice[:, :, z_idx]
        X, Y = cylindrical_to_cartesian(r, theta)
        
        im2 = axes[idx, 1].pcolormesh(X, Y, rtheta_slice, cmap='hot', shading='auto')
        axes[idx, 1].set_xlabel('X [m]')
        axes[idx, 1].set_ylabel('Y [m]')
        axes[idx, 1].set_title(f'r-θ cross-section (z={z[z_idx]:.3f}m), t={t_idx*dt:.2f}s')
        axes[idx, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[idx, 1])
        
        # θ-z cross-section (r=midpoint)
        r_idx = len(r) // 2
        thetaz_slice = T_slice[r_idx, :, :]
        Theta_grid, Z_grid = np.meshgrid(theta, z, indexing='ij')
        
        im3 = axes[idx, 2].pcolormesh(Theta_grid, Z_grid, thetaz_slice, cmap='hot', shading='auto')
        axes[idx, 2].set_xlabel('θ [rad]')
        axes[idx, 2].set_ylabel('z [m]')
        axes[idx, 2].set_title(f'θ-z cross-section (r={r[r_idx]:.1f}m), t={t_idx*dt:.2f}s')
        plt.colorbar(im3, ax=axes[idx, 2])
    
    plt.tight_layout()
    plt.show()


def plot_heatmap_grid(T, r, theta, z, z_levels, time_index, title='Temperature at Different Depths'):
    """
    Grid of 2D heatmaps at different z-depths.
    
    Parameters:
    -----------
    T : ndarray or list
        Temperature field(s) (Nt, Nr, Ntheta, Nz) or (Nr, Ntheta, Nz)
    r : ndarray
        Radial coordinates
    theta : ndarray
        Angular coordinates
    z : ndarray
        Axial coordinates
    z_levels : list
        z-indices or z-values to plot
    time_index : int
        Time index to plot
    title : str
        Plot title
    """
    if isinstance(T, list):
        T = np.array(T)
    
    # If single time step, add dimension
    if T.ndim == 3:
        T_slice = T
    else:
        T_slice = T[time_index]
    
    n_levels = len(z_levels)
    n_cols = 3
    n_rows = (n_levels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_levels == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    X, Y = cylindrical_to_cartesian(r, theta)
    vmin = np.min(T_slice)
    vmax = np.max(T_slice)
    
    for idx, z_level in enumerate(z_levels):
        if isinstance(z_level, float):
            # Find closest z index
            z_idx = np.argmin(np.abs(z - z_level))
        else:
            z_idx = z_level
        
        z_slice = T_slice[:, :, z_idx]
        
        im = axes[idx].pcolormesh(X, Y, z_slice, cmap='hot', shading='auto',
                                 vmin=vmin, vmax=vmax)
        axes[idx].set_xlabel('X [m]')
        axes[idx].set_ylabel('Y [m]')
        axes[idx].set_title(f'z = {z[z_idx]:.4f} m')
        axes[idx].set_aspect('equal')
        plt.colorbar(im, ax=axes[idx], label='Temperature')
    
    # Hide unused subplots
    for idx in range(n_levels, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_interactive_3d(T, r, theta, z, time_index):
    """
    Interactive, rotatable 3D visualization.
    
    Parameters:
    -----------
    T : ndarray or list
        Temperature field(s) (Nt, Nr, Ntheta, Nz) or (Nr, Ntheta, Nz)
    r : ndarray
        Radial coordinates
    theta : ndarray
        Angular coordinates
    z : ndarray
        Axial coordinates
    time_index : int
        Time index to plot
    """
    if isinstance(T, list):
        T = np.array(T)
    
    # If single time step, use it directly
    if T.ndim == 3:
        T_slice = T
    else:
        T_slice = T[time_index]
    
    try:
        import plotly.graph_objects as go
        
        # Convert to Cartesian
        R_grid, Theta_grid, Z_grid = np.meshgrid(r, theta, z, indexing='ij')
        X = R_grid * np.cos(Theta_grid)
        Y = R_grid * np.sin(Theta_grid)
        
        # Sample points for performance (every nth point)
        step = max(1, len(r) // 20)
        X_sample = X[::step, ::step, ::step]
        Y_sample = Y[::step, ::step, ::step]
        Z_sample = Z_grid[::step, ::step, ::step]
        T_sample = T_slice[::step, ::step, ::step]
        
        fig = go.Figure(data=go.Scatter3d(
            x=X_sample.flatten(),
            y=Y_sample.flatten(),
            z=Z_sample.flatten(),
            mode='markers',
            marker=dict(
                size=3,
                color=T_sample.flatten(),
                colorscale='Hot',
                showscale=True,
                colorbar=dict(title='Temperature')
            )
        ))
        
        fig.update_layout(
            title=f'Interactive 3D Temperature Field (t={time_index})',
            scene=dict(
                xaxis_title='X [m]',
                yaxis_title='Y [m]',
                zaxis_title='Z [m]'
            )
        )
        fig.show()
    except ImportError:
        # Fallback to matplotlib
        print("Plotly not available, using matplotlib 3D")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample for performance
        step = max(1, len(r) // 10)
        R_grid, Theta_grid, Z_grid = np.meshgrid(r[::step], theta[::step], z[::step], indexing='ij')
        X = R_grid * np.cos(Theta_grid)
        Y = R_grid * np.sin(Theta_grid)
        T_sample = T_slice[::step, ::step, ::step]
        
        scatter = ax.scatter(X.flatten(), Y.flatten(), Z_grid.flatten(),
                           c=T_sample.flatten(), cmap='hot', s=1)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(f'3D Temperature Field (t={time_index})')
        plt.colorbar(scatter, ax=ax, label='Temperature')
        plt.show()

