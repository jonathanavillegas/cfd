"""
2D visualization functions for heat transfer simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ..utils.coordinate_utils import cylindrical_to_cartesian


def plot_2d_heatmap(T, x, y, title='Temperature Distribution', cmap='hot'):
    """
    Plot 2D heatmap of temperature field.
    
    Parameters:
    -----------
    T : ndarray
        Temperature field (2D array)
    x, y : ndarray
        Spatial coordinates
    title : str
        Plot title
    cmap : str
        Colormap name
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(T, extent=[0, x[-1], 0, y[-1]], origin='lower', cmap=cmap)
    plt.colorbar(label='Temperature')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    plt.show()


def plot_2d_animated(T_frames, x, y, dt, title='Temperature Evolution', cmap='hot'):
    """
    Animated 2D heatmap showing temperature evolution.
    
    Parameters:
    -----------
    T_frames : list or ndarray
        List of temperature fields at each time step
    x, y : ndarray
        Spatial coordinates
    dt : float
        Time step
    title : str
        Plot title
    cmap : str
        Colormap name
    """
    T_frames = np.array(T_frames)
    fig, ax = plt.subplots()
    cax = ax.imshow(T_frames[0], cmap=cmap, origin='lower', 
                    extent=[0, x[-1], 0, y[-1]], 
                    vmin=np.min(T_frames), vmax=np.max(T_frames))
    fig.colorbar(cax, label='Temperature')
    ax.set_title(f"{title} - Time = 0.0 s")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    def update(frame):
        cax.set_array(T_frames[frame])
        ax.set_title(f"{title} - Time = {frame * dt:.2f} s")
        return [cax]

    ani = animation.FuncAnimation(
        fig, update, frames=T_frames.shape[0], interval=10, blit=False
    )
    plt.show()


def plot_cylindrical_2d_animated(T_frames, r, theta, dt, title='Temperature Evolution', cmap='hot'):
    """
    Animated 2D plot for cylindrical coordinates.
    
    Parameters:
    -----------
    T_frames : list or ndarray
        List of temperature fields at each time step (Nr, Ntheta)
    r : ndarray
        Radial coordinates
    theta : ndarray
        Angular coordinates
    dt : float
        Time step
    title : str
        Plot title
    cmap : str
        Colormap name
    """
    T_frames = np.array(T_frames)
    X, Y = cylindrical_to_cartesian(r, theta)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    c = ax.pcolormesh(X, Y, T_frames[0], shading='auto', cmap=cmap, 
                     vmin=np.min(T_frames), vmax=np.max(T_frames))
    fig.colorbar(c, ax=ax)
    ax.set_title(f'{title} - Time = 0.0')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')

    def animate(frame):
        c.set_array(T_frames[frame].flatten())
        ax.set_title(f'{title} - Time: {frame * dt:.2f}')
        return c,

    ani = animation.FuncAnimation(fig, animate, frames=len(T_frames), interval=dt*1000)
    plt.show()

