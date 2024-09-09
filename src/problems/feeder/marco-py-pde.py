from typing import List
import numpy as np
from pde import (
    FieldCollection,
    PDEBase,
    ScalarField,
    CartesianGrid,
    UnitGrid,
    MemoryStorage,
    movie,
)

# Define the size of the grid and create it
grid = CartesianGrid(
    [[0, 1]], [500], periodic=[False]
)  # 1D grid with 100 points, x ∈ [0, 1]


kernel = np.ones(5) / 5
disturbance = np.ones(110) * 0.0
disturbance[15:25] = 1.0
disturbance = np.convolve(disturbance, kernel, mode="same")[5:105]

# Define the initial condition for g(x, t)

ch = ScalarField(
    grid, 0.0
)  # Initialise the ScalarField for the concentration in the hopper
cc = ScalarField(
    grid, 0.0
)  # Initialise the ScalarField for the concentration in the bottom part
L = ScalarField(grid, 0.5)  # Initialisation of height in hopper
vc = ScalarField(grid, 0.001)  # Initialisation of powder volume in bottom part
omega = ScalarField(grid, 200.0 / 60.0)  # RPS screw
error = ScalarField(grid, 0.0)

# Define the PDE components
# v = 1.0  # example constant velocity, can be time-dependent
# D = 0.01  # example diffusion coefficient
# A = 1.0  # example cross-sectional area
V_out = lambda t: 0.005  # example outflow rate, can be time-dependent


class Feeder(PDEBase):
    def __init__(
        self, A: float, Vc: float, rho_bulk: float, mass_flowrate_setpoint: float
    ):
        self.A = A
        self.Vc = Vc
        self.rho_bulk = rho_bulk
        self.mass_flowrate_setpoint = mass_flowrate_setpoint
        self.state = "discharge"
        self.counter = None

        # Parameters for feed factor model
        self.ff_max = 1.54 * 10
        self.ff_min = 2.7 * 10
        self.beta = 0.01

    # Feed factor model function based on the exponential decay relationship
    def feed_factor_model(self, weight: float) -> float:
        return self.ff_max - (self.ff_max - self.ff_min) * np.exp(-self.beta * weight)

    # Screw Feeder Model: Flow rate calculation
    def screw_feeder_flow_rate(screw_speed: float, feed_factor: float) -> float:
        """Calculate flow rate from screw speed and feed factor."""
        return screw_speed * feed_factor

    # Define the PDE system
    def evolution_rate(self, state, t):
        ch = state[0]
        cc = state[1]
        L = state[2]
        omega = state[3]

        if self.state == "discharge" and L.average < 0.2:
            self.state = "charge"
            # The following counter handles the reset of the disturbance
            if not self.counter:
                self.counter = np.arange(0 + int(t), 100 + int(t), 1)
        if self.state == "charge" and L.average > 0.6:
            self.state = "discharge"
            self.counter = None

        if self.state == "charge":
            V_in = 0.001
        else:
            V_in = 0.0

        # Compute m_out
        weight = self.rho_bulk * (self.Vc + self.A * L.average)
        ff = self.feed_factor_model(weight=weight)
        m_out = omega * ff  # omega is rev/s ff is g/rev -> m_out is g/s
        m_out_si = m_out / 1000.0  # m_out_si is the mass flowrate in kg/s
        V_out = m_out_si / self.rho_bulk  # m3/s
        v_hopper = V_out / self.A

        L_rate = (V_in - V_out) / self.A
        # Update L(t)
        dL_dt = ScalarField(grid, L_rate)

        def boundary(value, dx, coords, t):
            if self.state == "charge":
                value = np.interp([t], self.counter, disturbance)[0]
                return value
            else:
                return value

        # Calculate the right-hand side of the PDE
        dch_dt = (
            -(1 / L)
            * (-v_hopper - grid.axes_coords[0] * dL_dt)
            * ch.gradient(
                bc=[
                    {"curvature": 0.0},
                    {"value_expression": boundary},
                ],
                args=dict(t=t),
            )
        )

        dcc_dt = (1 / self.Vc) * (V_out * ch.data[0] - V_out * cc)

        domega_dt = ScalarField(grid, 0.0)

        return FieldCollection([dch_dt, dcc_dt, dL_dt, domega_dt])


# Instantiate Equations
eq = Feeder(
    A=np.pi * 0.2**2,  # m2
    Vc=0.005,  # m3
    rho_bulk=712.0,  # kg/m3
    mass_flowrate_setpoint=18.0,  # kg/h
)

# Initial conditions for both g(x, t) and L(t)
state = FieldCollection([ch, cc, L, omega])

# Create the PDE object without specifying a solver
storage = MemoryStorage()

# Solve the PDE
result = eq.solve(
    state,
    t_range=1500.0,
    dt=0.01,
    tracker=["progress", storage.tracker(interrupts=10)],
    solver="ScipySolver",
    method="LSODA",
)

# Visualize the resu
movie(storage, filename="output.gif", plot_args=dict(ylim=(-1.0, 2.5)))