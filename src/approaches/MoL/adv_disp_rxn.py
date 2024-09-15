from juliacall import Main as jl
import plotly.graph_objects as go
import numpy as np
import time
from pydantic import BaseModel
from typing import Literal
import mlflow

from utils.visualisation import create_gif
from utils.optuna_utils import run_optuna_study
from utils.sweep_utils import parameter_sweep

from approaches.MoL.adv_disp import AdvDispParams  # Import AdvDispParams

class AdvDispRxnParams(AdvDispParams):  # Inherit from AdvDispParams
    reaction_rate: float = 0.01  # Added reaction rate parameter

# Include multiple Julia scripts
jl.include("src/approaches/MoL/adv_disp_rxn.jl")

def run(config):
    params = AdvDispRxnParams(
        v=config["v"],
        DaxialA=config["DaxialA"],
        DaxialB=config["DaxialB"],
        DaxialC=config["DaxialC"],
        L=config["L"],
        tf=config["tf"],
        solver=config["solver_params"]["solver"],
        method=config["solver_params"]["method"],
        grid_size=config["grid_size"],
        reaction_rate=config["reaction_rate"]
    )

    mlflow.set_experiment(config["experiment_id"])

    # # End any active run before starting a new one
    # if mlflow.active_run() is not None:
    #     mlflow.end_run()

    with mlflow.start_run():
        log_adv_disp_params(config)
        log_adv_disp_tags(config["tags"])

        print("Starting pde solving...")
        start_time = time.time()
        
        solution_array_A_julia, solution_array_B_julia, solution_array_C_julia = jl.adv_disp_rxn_all(params.model_dump())

        end_solution_time = time.time()
        
        # convert julia arrays to numpy arrays
        solution_array_A = solution_array_A_julia.to_numpy()
        solution_array_B = solution_array_B_julia.to_numpy()
        solution_array_C = solution_array_C_julia.to_numpy()
        
        end_conversion_time = time.time()
        
        print(f"adv_disp_rxn_pulse completed in {end_solution_time - start_time:.2f} seconds.")
        print(f"Conversion completed in {end_conversion_time - end_solution_time:.2f} seconds.")

        # Log metrics
        log_adv_disp_metrics(end_solution_time - start_time, end_conversion_time - end_solution_time)

        # Create a .gif movie of the solutions
        print("Creating .gif movies...")
        create_gif([solution_array_A, solution_array_B, solution_array_C], 
                   ['Solution A', 'Solution B', 'Solution C'], 
                   config["output_filename"])
        print(f"Created {config['output_filename']}")

        # Log the .gif movie to MLflow
        print("Logging .gif movies to MLflow...")
        mlflow.log_artifact(config["output_filename"])
        print(f"Logged {config['output_filename']} to MLflow")

def log_adv_disp_tags(tags):
    """Log adv_disp tags to MLflow."""
    for key, value in tags.items():
        mlflow.set_tag(key, value)

def log_adv_disp_params(config):
    """Log adv_disp parameters to MLflow."""
    mlflow.log_params(config)

def log_adv_disp_metrics(solution_time, conversion_time):
    """Log adv_disp metrics to MLflow."""
    mlflow.log_metric("solution_time", solution_time)
    mlflow.log_metric("conversion_time", conversion_time)
    mlflow.log_metric("execution_time", solution_time + conversion_time)

def single_run():
    config = {
        "v": 0.10,
        "DaxialA": 1e-2,
        "DaxialB": 1e-4,
        "DaxialC": 1e-10,
        "L": 1.0,
        "tf": 10.0,
        "solver_params": {
            "solver": "MethodOfLines.jl",
            "method": "Tsit5"
        },
        "grid_size": 100,
        "reaction_rate": 1,
        "output_filename": "solutions.gif",
        "experiment_id": "adv_disp_rxn",
        "tags": {
            "problem": "adv_disp_rxn",
            "dimension": "1D",
            "num_components": 3,
            "num_phases": 1,
            "run_type": "single_run",
            "inlet_profile": "all"
        }
    }
    run(config)

def identify_best_method():
    # Consolidated configuration dictionary for hyperparameters and Optuna study
    optuna_config = {
        "v": 0.10,
        "DaxialA": 1e-2,
        "DaxialB": 1e-4,
        "DaxialC": 1e-10,
        "L": 1.0,
        "tf": 10.0,
        "grid_size": 200,
        "reaction_rate": 0.01,
        "output_filename": "solutions.gif",
        "experiment_id": "adv_disp_rxn",
        "tags": {
            "problem": "adv_disp_rxn",
            "dimension": "1D",
            "num_components": 3,
            "num_phases": 1,
            "run_type": "identify_best_method"  # Added run_type tag
        },
        "solver_params": {
            "solver": "MethodOfLines.jl",
            "method": None  # Initialize method key
        },
        "solvers": {
            "MethodOfLines.jl": {
                "methods": ["Tsit5", "RK4", "Vern7", "DP5", "BS3", "Rodas5", "TRBDF2", "KenCarp4", "Euler", "ImplicitEuler", "Heun"]
            }
        },
        "timeout": 40,  # Set a timeout to whatever is time to give up on a solver
        "run_name_prefix": "adv_disp_rxn",
        "pde_package": "julia",
        "parent_run": "adv_disp_rxn_parent"
    }

    # Run the Optuna study
    run_optuna_study(optuna_config, run, log_adv_disp_params, n_trials=11)

def find_grid_size_for_timeout():
    optuna_config = {
        "v": 0.10,
        "DaxialA": 1e-2,
        "DaxialB": 1e-4,
        "DaxialC": 1e-10,
        "L": 1.0,
        "tf": 10.0,
        "grid_size": 200,
        "reaction_rate": 0.01,
        "output_filename": "solutions.gif",
        "experiment_id": "adv_disp_rxn",
        "tags": {
            "problem": "adv_disp_rxn",
            "dimension": "1D",
            "num_components": 3,
            "num_phases": 1,
            "run_type": "find_grid_size_for_timeout"  # Added run_type tag
        },
        "solver_params": {
            "solver": "MethodOfLines.jl",
            "method": None  # Initialize method key
        },
        "solvers": {
            "MethodOfLines.jl": {
                "methods": ["Tsit5", "RK4", "Vern7", "DP5", "BS3", "Rodas5", "TRBDF2", "KenCarp4", "Euler", "ImplicitEuler", "Heun"]
            }
        },
        "timeout": 10,  # Set a timeout to whatever is time to give up on a solver
        "run_name_prefix": "adv_disp_rxn",
        "pde_package": "julia",
        "parent_run": "adv_disp_rxn_parent"
    }

    timeout = 10  # Example timeout value in seconds
    parameter_sweep(optuna_config, run, log_adv_disp_params, timeout)

if __name__ == "__main__":
    single_run()  # Uncomment to run a single run
    # identify_best_method()  # Uncomment to do a study to find the best method
    # find_grid_size_for_timeout()  # Uncomment to perform the parameter sweep