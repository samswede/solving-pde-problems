from juliacall import Main as jl
import plotly.graph_objects as go
import numpy as np
import time
# from utils.visualisation import create_gif

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
    ani.save(filename, writer='imagemagick', fps=10)
    plt.close()


# Include multiple Julia scripts
jl.include("src/approaches/MoL/adv_disp.jl")

if __name__ == "__main__":
    v = 0.15
    C_in = 1.0
    Daxial = 1e-3
    L = 1.0
    tf = 10.0

    # Choose which function to call
    function_choice = "solve_adv_disp_3"  # Change this to "solve_adv_disp_2" or "solve_adv_disp_3" as needed

    if function_choice in ["solve_adv_disp", "solve_adv_disp_2"]:
        if function_choice == "solve_adv_disp":
            solution_array = jl.solve_adv_disp(v, C_in, Daxial, L, tf).to_numpy()
        else:
            solution_array = jl.solve_adv_disp_2(v, Daxial, L, tf).to_numpy()

        # Plot the solution using Plotly
        x = np.linspace(0, 1, solution_array.shape[1])
        y = np.linspace(0, 10, solution_array.shape[0])
        X, Y = np.meshgrid(x, y)

        fig = go.Figure(data=[go.Surface(z=solution_array, x=X, y=Y)])

        fig.update_layout(
            title='Solution Array 3D Surface',
            scene=dict(
                xaxis_title='Z (Position)',
                yaxis_title='T (Time)',
                zaxis_title='C (Concentration)'
            )
        )

        fig.show()

    else:  # solve_adv_diff_3

        print("Starting solve_adv_disp_3...")
        start_time = time.time()
        
        solution_array_A_julia, solution_array_B_julia, solution_array_C_julia = jl.solve_adv_disp_3()

        # convert julia arrays to numpy arrays
        solution_array_A = solution_array_A_julia.to_numpy()
        solution_array_B = solution_array_B_julia.to_numpy()
        solution_array_C = solution_array_C_julia.to_numpy()
        
        end_time = time.time()
        print(f"solve_adv_disp_3 completed in {end_time - start_time:.2f} seconds.")

        # Create a .gif movie of the solutions
        print("Creating .gif movies...")
        create_gif([solution_array_A, solution_array_B, solution_array_C], 
                   ['Solution A', 'Solution B', 'Solution C'], 
                   'solutions.gif')
        print("Created solutions.gif")