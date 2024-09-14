import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_gif(solution_array, filename='solution_animation.gif'):
    """
    Create a .gif movie from the solution array.

    Parameters:
    - solution_array: 2D numpy array where rows represent time steps and columns represent spatial positions.
    - filename: Name of the output .gif file.
    """
    fig, ax = plt.subplots()
    line, = ax.plot(solution_array[0, :])
    ax.set_xlim(0, solution_array.shape[1] - 1)
    ax.set_ylim(np.min(solution_array), np.max(solution_array))
    ax.set_xlabel('Position')
    ax.set_ylabel('Concentration')

    def update(frame):
        line.set_ydata(solution_array[frame, :])
        ax.set_title(f'Time step {frame}')
        return line,

    ani = FuncAnimation(fig, update, frames=range(solution_array.shape[0]), blit=True)
    ani.save(filename, writer='imagemagick', fps=10)
    plt.close()

# Example usage:
# create_gif(solution_array_A)
