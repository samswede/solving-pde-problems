from pydantic import BaseModel
from typing import Tuple
import numpy as np
from pde import (
    FieldCollection,
    PDEBase,
    ScalarField,
    CartesianGrid,
    MemoryStorage,
    movie,
)
import mlflow
import time

from utils.optuna_utils import run_optuna_study
from utils.sweep_utils import parameter_sweep

class FeederParams(BaseModel):
    A: float
    Vc: float
    rho_bulk: float
    mass_flowrate_setpoint: float

class InitialConditions(BaseModel):
    ch: float
    cc: float
    L: float
    vc: float
    omega: float
    error: float

class DisturbanceParams(BaseModel):
    range: Tuple[int, int]
    value: float
    kernel_size: int

class SolverParams(BaseModel):
    t_range: float
    dt: float
    solver: str
    method: str

class FeederConfig(BaseModel):
    grid_size: int
    initial_conditions: InitialConditions
    feeder_params: FeederParams
    disturbance_params: DisturbanceParams
    solver_params: SolverParams
    output_filename: str

class Feeder(PDEBase):
    def __init__(self, feeder_params: FeederParams, initial_conditions: InitialConditions, grid_size: int):
        self.A = feeder_params.A
        self.Vc = feeder_params.Vc
        self.rho_bulk = feeder_params.rho_bulk
        self.mass_flowrate_setpoint = feeder_params.mass_flowrate_setpoint
        self.state = "discharge"
        self.counter = None

        # Parameters for feed factor model
        self.ff_max = 1.54 * 10
        self.ff_min = 2.7 * 10
        self.beta = 0.01

        # Initialize the grid
        self.grid = CartesianGrid([[0, 1]], [grid_size], periodic=[False])

        # Initialize the initial conditions
        self.ch = ScalarField(self.grid, initial_conditions.ch)
        self.cc = ScalarField(self.grid, initial_conditions.cc)
        self.L = ScalarField(self.grid, initial_conditions.L)
        self.vc = ScalarField(self.grid, initial_conditions.vc)
        self.omega = ScalarField(self.grid, initial_conditions.omega)
        self.error = ScalarField(self.grid, initial_conditions.error)

        # Default disturbance
        self.disturbance = np.zeros(100)  # Example default disturbance

    def set_disturbance(self, disturbance: np.ndarray):
        """Set a new disturbance array."""
        self.disturbance = disturbance

    def feed_factor_model(self, weight: float) -> float:
        return self.ff_max - (self.ff_max - self.ff_min) * np.exp(-self.beta * weight)

    def screw_feeder_flow_rate(screw_speed: float, feed_factor: float) -> float:
        return screw_speed * feed_factor

    def evolution_rate(self, state, t):
        ch = state[0]
        cc = state[1]
        L = state[2]
        omega = state[3]

        if self.state == "discharge" and L.average < 0.2:
            self.state = "charge"
            if not self.counter:
                self.counter = np.arange(0 + int(t), 100 + int(t), 1)
        if self.state == "charge" and L.average > 0.6:
            self.state = "discharge"
            self.counter = None

        if self.state == "charge":
            V_in = 0.001
        else:
            V_in = 0.0

        weight = self.rho_bulk * (self.Vc + self.A * L.average)
        ff = self.feed_factor_model(weight=weight)
        m_out = omega * ff
        m_out_si = m_out / 1000.0
        V_out = m_out_si / self.rho_bulk
        v_hopper = V_out / self.A

        L_rate = (V_in - V_out) / self.A
        dL_dt = ScalarField(self.grid, L_rate)

        def boundary(value, dx, coords, t):
            if self.state == "charge":
                value = np.interp([t], self.counter, self.disturbance)[0]
                return value
            else:
                return value

        dch_dt = (
            -(1 / L)
            * (-v_hopper - self.grid.axes_coords[0] * dL_dt)
            * ch.gradient(
                bc=[
                    {"curvature": 0.0},
                    {"value_expression": boundary},
                ],
                args=dict(t=t),
            )
        )

        dcc_dt = (1 / self.Vc) * (V_out * ch.data[0] - V_out * cc)
        domega_dt = ScalarField(self.grid, 0.0)

        return FieldCollection([dch_dt, dcc_dt, dL_dt, domega_dt])

def run(config):
    # Define the size of the grid and create it
    grid_size = config["grid_size"]
    initial_conditions = InitialConditions(**config["initial_conditions"])
    feeder_params = FeederParams(**config["feeder_params"])
    disturbance_params = DisturbanceParams(**config["disturbance_params"])

    # Instantiate Equations
    eq = Feeder(
        feeder_params=feeder_params,
        initial_conditions=initial_conditions,
        grid_size=grid_size
    )

    # Set a new disturbance if needed
    disturbance = np.ones(110) * 0.0
    disturbance[disturbance_params.range[0]:disturbance_params.range[1]] = disturbance_params.value
    disturbance = np.convolve(disturbance, np.ones(disturbance_params.kernel_size) / disturbance_params.kernel_size, mode="same")[5:105]
    eq.set_disturbance(disturbance)

    # Initial conditions for both g(x, t) and L(t)
    state = FieldCollection([eq.ch, eq.cc, eq.L, eq.omega])

    # Create the PDE object without specifying a solver
    storage = MemoryStorage()

    # Record the start time
    start_time = time.time()

    # Solve the PDE
    result = eq.solve(
        state,
        t_range=config["solver_params"]["t_range"],
        dt=config["solver_params"]["dt"],
        tracker=["progress", storage.tracker(interrupts=10)],
        solver=config["solver_params"]["solver"],
        method=config["solver_params"]["method"],
    )

    # Record the end time
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Log metrics
    log_feeder_metrics(eq, execution_time)

    # Visualize the result and log artifact
    # output_filename = f"advection_diffusion_{config['grid_size']}_{disturbance_params.kernel_size}.gif"
    # movie(storage, filename=output_filename, plot_args=dict(ylim=(-1.0, 2.5)))
    # mlflow.log_artifact(output_filename)

def log_feeder_params(config):
    """Log feeder parameters to MLflow."""
    mlflow.log_params({
        "grid_size": config["grid_size"],
        "initial_conditions": config["initial_conditions"],
        "feeder_params": config["feeder_params"],
        "disturbance_params": config["disturbance_params"],
        "solver_params": config["solver_params"]
    })

def log_feeder_metrics(eq, execution_time):
    """Log feeder metrics to MLflow."""
    mlflow.log_metric("final_ch", eq.ch.data.mean())
    mlflow.log_metric("final_cc", eq.cc.data.mean())
    mlflow.log_metric("final_L", eq.L.data.mean())
    mlflow.log_metric("final_omega", eq.omega.data.mean())
    mlflow.log_metric("execution_time", execution_time)

def single_run():
    # just config for a single run
    config = {
        "grid_size": 200,
        "initial_conditions": {
            "ch": 0.0,
            "cc": 0.0,
            "L": 0.5,
            "vc": 0.001,
            "omega": 200.0 / 60.0,
            "error": 0.0,
        },
        "feeder_params": {
            "A": np.pi * 0.2**2,
            "Vc": 0.005,
            "rho_bulk": 712.0,
            "mass_flowrate_setpoint": 18.0,
        },
        "disturbance_params": {
            "range": (15, 25),
            "value": 1.0,
            "kernel_size": 5,
        },
        "solver_params": {
            "t_range": 1500.0,
            "dt": 0.01,
            "solver": "ScipySolver",
            "method": "RK45",
        },
        
        "output_filename": "output.gif",
    }
    run(config)


def identify_best_method():
    # Consolidated configuration dictionary for hyperparameters and Optuna study
    optuna_config = {
        "grid_size": 200,
        "initial_conditions": {
            "ch": 0.0,
            "cc": 0.0,
            "L": 0.5,
            "vc": 0.001,
            "omega": 200.0 / 60.0,
            "error": 0.0,
        },
        "feeder_params": {
            "A": np.pi * 0.2**2,
            "Vc": 0.005,
            "rho_bulk": 712.0,
            "mass_flowrate_setpoint": 18.0,
        },
        "disturbance_params": {
            "range": (15, 25),
            "value": 1.0,
            "kernel_size": 5,
        },
        "solver_params": {
            "t_range": 1500.0,
            "dt": 0.01,
        },
        "output_filename": "output.gif",
        "solvers": {
            "ScipySolver": {
                "methods": ["LSODA", "RK45", "RK23", "Radau", "BDF"]
            },
            # "AnotherSolver": {
            #     "methods": ["Method1", "Method2", "Method3"]
            # }
        },
        "timeout": 10,  # Set a timeout to whatever is time to give up on a solver
        "run_name_prefix": "advection_diffusion_2",
        "pde_package": "py-pde",
        "parent_run": "advection_diffusion_2_parent"
    }

    # Run the Optuna study
    run_optuna_study(optuna_config, run, log_feeder_params, n_trials=5)

def find_grid_size_for_timeout():
    optuna_config = {
        "grid_size": 200,
        "initial_conditions": {
            "ch": 0.0,
            "cc": 0.0,
            "L": 0.5,
            "vc": 0.001,
            "omega": 200.0 / 60.0,
            "error": 0.0,
        },
        "feeder_params": {
            "A": np.pi * 0.2**2,
            "Vc": 0.005,
            "rho_bulk": 712.0,
            "mass_flowrate_setpoint": 18.0,
        },
        "disturbance_params": {
            "range": (15, 25),
            "value": 1.0,
            "kernel_size": 5,
        },
        "solver_params": {
            "t_range": 1500.0,
            "dt": 0.01,
        },
        "output_filename": "output.gif",
        "solvers": {
            "ScipySolver": {
                "methods": ["LSODA", "RK45", "RK23", "Radau", "BDF"]
            },
            # "AnotherSolver": {
            #     "methods": ["Method1", "Method2", "Method3"]
            # }
        },
        "timeout": 10,  # Set a timeout to whatever is time to give up on a solver
        "run_name_prefix": "advection_diffusion_2",
        "pde_package": "py-pde",
        "parent_run": "advection_diffusion_2_parent"
    }

    timeout = 10  # Example timeout value in seconds
    parameter_sweep(optuna_config, run, log_feeder_params, timeout)

if __name__ == "__main__":
    # identify_best_method()  # Run this to do a study to find the best method
    find_grid_size_for_timeout()  # Run this to perform the parameter sweep