using DifferentialEquations

function solve_simple_ode(x)
    # Solving the ODE:
    # du/dt = 1.01u
    # u(0) = 1.0
    # t ∈ [0, x]
    
    ode = ODEProblem((u,p,t) -> 1.01u, 1.0, (0.0, x))
    solution = solve(ode)
    
    # Return an array of the variables (time and u)
    return [solution.t solution.u]
end