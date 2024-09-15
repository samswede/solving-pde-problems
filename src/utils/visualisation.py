import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_gif(solution_arrays, labels, filename='solution_animation.gif'):
    """
    Create a .gif movie from the list of solution arrays.

    Parameters:
    - solution_arrays: List of 2D numpy arrays where rows represent time steps and columns represent spatial positions.
    - labels: List of labels for each solution array.
    - filename: Name of the output .gif file.
    """
    fig, ax = plt.subplots()
    lines = [ax.plot(solution_array[0, :], label=label)[0] for solution_array, label in zip(solution_arrays, labels)]
    ax.set_xlim(0, solution_arrays[0].shape[1] - 1)
    ax.set_ylim(min(np.min(solution_array) for solution_array in solution_arrays), 
                max(np.max(solution_array) for solution_array in solution_arrays))
    ax.set_xlabel('Position')
    ax.set_ylabel('Concentration')
    ax.legend()

    def update(frame):
        for line, solution_array in zip(lines, solution_arrays):
            line.set_ydata(solution_array[frame, :])
        ax.set_title(f'Time step {frame}')
        return lines

    ani = FuncAnimation(fig, update, frames=range(solution_arrays[0].shape[0]), blit=True)
    # ani.save(filename, writer='imagemagick', fps=10)
    ani.save(filename, writer='pillow', fps=10)
    plt.close()

# Example usage:
# create_gif(solution_array_A)
