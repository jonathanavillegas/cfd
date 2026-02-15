"""
2D visualization functions for heat transfer simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ..utils.coordinate_utils import cylindrical_to_cartesian


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
                    extent=(0, x[-1], 0, y[-1]), 
                    vmin=np.min(T_frames), vmax=np.max(T_frames))
    fig.colorbar(cax, label='Temperature [K]')
    ax.set_title(f"{title} - Time = 0.0 s")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    def update(frame):
        cax.set_array(T_frames[frame])
        ax.set_title(f"{title} - Time = {frame * dt:.2f} s")
        return [cax]

    ani = animation.FuncAnimation(
        fig, update, frames=T_frames.shape[0], interval=50, blit=False
    )
    plt.show()


def save_2d_animated_gif(T_frames, x, y, dt, filename='temperature_evolution.gif', 
                         title='Temperature Evolution', cmap='hot', fps=20):
    """
    Save animated 2D heatmap as GIF showing temperature evolution.
    
    Parameters:
    -----------
    T_frames : list or ndarray
        List of temperature fields at each time step
    x, y : ndarray
        Spatial coordinates
    dt : float
        Time step
    filename : str
        Output filename for GIF (default: 'temperature_evolution.gif')
    title : str
        Plot title
    cmap : str
        Colormap name
    fps : int
        Frames per second for GIF (default: 20)
    """
    T_frames = np.array(T_frames)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(T_frames[0], cmap=cmap, origin='lower', 
                    extent=(0, x[-1], 0, y[-1]), 
                    vmin=np.min(T_frames), vmax=np.max(T_frames))
    cbar = fig.colorbar(cax, label='Temperature [K]')
    ax.set_title(f"{title} - Time = 0.0 s")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    def update(frame):
        cax.set_array(T_frames[frame])
        ax.set_title(f"{title} - Time = {frame * dt:.2f} s")
        return [cax]

    ani = animation.FuncAnimation(
        fig, update, frames=T_frames.shape[0], interval=1000/fps, blit=False
    )
    
    # Save as GIF
    print(f"Saving animation to {filename}...")
    ani.save(filename, writer='pillow', fps=fps)
    print(f"Animation saved successfully!")
    plt.close(fig)


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
    coords = cylindrical_to_cartesian(r, theta)
    X, Y = coords[0], coords[1]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    c = ax.pcolormesh(X, Y, T_frames[0], shading='auto', cmap=cmap, 
                     vmin=np.min(T_frames), vmax=np.max(T_frames))
    fig.colorbar(c, ax=ax, label='Temperature [K]')
    ax.set_title(f'{title} - Time = 0.0 s')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')

    def animate(frame):
        c.set_array(T_frames[frame].flatten())
        ax.set_title(f'{title} - Time: {frame * dt:.2f} s')
        return c,

    ani = animation.FuncAnimation(fig, animate, frames=len(T_frames), interval=dt*1000*2)
    plt.show()

