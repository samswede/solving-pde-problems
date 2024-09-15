from juliacall import Main as jl
import plotly.graph_objects as go
import numpy as np
import time
from pydantic import BaseModel
from typing import Literal

from utils.visualisation import create_gif

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



class AdvDispParams(BaseModel):
    v: float = 0.10
    DaxialA: float = 1e-2
    DaxialB: float = 1e-4
    DaxialC: float = 1e-10
    L: float = 1.0
    tf: float = 10.0
    solver_name: Literal[
        "Tsit5", "RK4", "Vern7", "DP5", "BS3",
        "Rodas5", "TRBDF2", "KenCarp4",
        "Euler", "ImplicitEuler", "Heun",
    ] = "Tsit5"
    num_elements: int = 1000

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


# Include multiple Julia scripts
jl.include("src/approaches/MoL/adv_disp.jl")

if __name__ == "__main__":
    params = AdvDispParams(
        v=0.10,
        DaxialA=1e-2,
        DaxialB=1e-4,
        DaxialC=1e-10,
        L=1.0,
        tf=10.0,
        solver_name="Rodas5",
        num_elements=200
    )

    print("Starting pde solving...")
    start_time = time.time()
    
    solution_array_A_julia, solution_array_B_julia, solution_array_C_julia = jl.solve_adv_disp_pulse(params.model_dump())

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