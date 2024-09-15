using ModelingToolkit
using OrdinaryDiffEq
using MethodOfLines, DomainSets
using Interpolations
using PythonCall

# Helper function to map solver names to solver objects
function get_solver(solver_name::String)
    solvers = Dict(
        "Tsit5" => Tsit5(),
        "RK4" => RK4(),
        "Vern7" => Vern7(),
        "DP5" => DP5(),
        "BS3" => BS3(),
        "Rodas5" => Rodas5(),
        "TRBDF2" => TRBDF2(),
        "KenCarp4" => KenCarp4(),
        "Euler" => Euler(),
        "ImplicitEuler" => ImplicitEuler(),
        "Heun" => Heun(),
    )
    return solvers[solver_name]
end

function adv_disp_eqrxn_pulse(params::PyDict)
    v = params["v"]
    DaxialA = params["DaxialA"]
    DaxialB = params["DaxialB"]
    DaxialC = params["DaxialC"]
    L = params["L"]
    tf = params["tf"]
    solver_name = params["method"]
    grid_size = params["grid_size"]
    K_eq = params["equilibrium_constant"]

    solver = get_solver(solver_name)

    @parameters t, z
    @variables CA(..) CB(..) CC(..)
    Dt = Differential(t)
    Dz = Differential(z)
    Dzz = Differential(z)^2

    # define the pulse function
    start = 0.5
    duration = 1.0
    amplitude = 0.5
    end_time = start + duration

    C_in_A(t) = amplitude * (tanh(100 * (t - start)) - tanh(100 * (t - end_time)))
    C_in_B(t) = amplitude * (tanh(100 * (t - start)) - tanh(100 * (t - end_time)))
    C_in_C(t) = 0

    eqA = Dt(CA(t, z)) ~ DaxialA * Dzz(CA(t, z)) - v * Dz(CA(t, z)) - K_eq * CA(t, z) * CB(t, z)
    eqB = Dt(CB(t, z)) ~ DaxialB * Dzz(CB(t, z)) - v * Dz(CB(t, z)) - K_eq * CA(t, z) * CB(t, z)
    eqC = Dt(CC(t, z)) ~ DaxialC * Dzz(CC(t, z)) - v * Dz(CC(t, z)) + K_eq * CA(t, z) * CB(t, z)
    eqR = CA(t, z) ~ K_eq * CB(t, z)

    ic_bc = [
        Dz(CA(t, 0)) ~ v * (CA(t, 0) - C_in_A(t)) / DaxialA,
        Dz(CA(t, L)) ~ 0,
        CA(0, z) ~ 1e-2,

        Dz(CB(t, 0)) ~ v * (CB(t, 0) - C_in_B(t)) / DaxialB,
        Dz(CB(t, L)) ~ 0,
        CB(0, z) ~ 1e-2,

        Dz(CC(t, 0)) ~ v * (CC(t, 0) - C_in_C(t)) / DaxialC,
        Dz(CC(t, L)) ~ 0,
        CC(0, z) ~ 1e-2
    ]

    domains = [t ∈ (0.0, tf), z ∈ (0.0, L)]

    @named pde_system = PDESystem([eqA, eqB, eqC, eqR], ic_bc, domains, [t, z], [CA(t, z), CB(t, z), CC(t, z)])

    dz = L / grid_size
    discretization = MOLFiniteDifference([z => dz], t, use_ODAE=true)

    prob = discretize(pde_system, discretization)

    sol = solve(prob, solver, saveat=0.2)

    # Convert the solutions to 2D arrays
    solution_array_A = Array(sol[CA(t, z)])
    solution_array_B = Array(sol[CB(t, z)])
    solution_array_C = Array(sol[CC(t, z)])

    return solution_array_A, solution_array_B, solution_array_C
end

function adv_disp_eqrxn_wave(params::PyDict)
    v = params["v"]
    DaxialA = params["DaxialA"]
    DaxialB = params["DaxialB"]
    DaxialC = params["DaxialC"]
    L = params["L"]
    tf = params["tf"]
    solver_name = params["method"]
    grid_size = params["grid_size"]
    K_eq = params["equilibrium_constant"]

    solver = get_solver(solver_name)

    @parameters t, z
    @variables CA(..) CB(..) CC(..)
    Dt = Differential(t)
    Dz = Differential(z)
    Dzz = Differential(z)^2

    C_in_A(t) = sin(t) + 1.1  # Ensure it's greater than 0
    C_in_B(t) = sin(t) + 1.1
    C_in_C(t) = 0

    eqA = Dt(CA(t, z)) ~ DaxialA * Dzz(CA(t, z)) - v * Dz(CA(t, z)) - K_eq * CA(t, z) * CB(t, z)
    eqB = Dt(CB(t, z)) ~ DaxialB * Dzz(CB(t, z)) - v * Dz(CB(t, z)) - K_eq * CA(t, z) * CB(t, z)
    eqC = Dt(CC(t, z)) ~ DaxialC * Dzz(CC(t, z)) - v * Dz(CC(t, z)) + K_eq * CA(t, z) * CB(t, z)
    eqR = CA(t, z) ~ K_eq * CB(t, z)

    ic_bc = [
        Dz(CA(t, 0)) ~ v * (CA(t, 0) - C_in_A(t)) / DaxialA,
        Dz(CA(t, L)) ~ 0,
        CA(0, z) ~ 1e-2,

        Dz(CB(t, 0)) ~ v * (CB(t, 0) - C_in_B(t)) / DaxialB,
        Dz(CB(t, L)) ~ 0,
        CB(0, z) ~ 1e-2,

        Dz(CC(t, 0)) ~ v * (CC(t, 0) - C_in_C(t)) / DaxialC,
        Dz(CC(t, L)) ~ 0,
        CC(0, z) ~ 1e-2
    ]

    domains = [t ∈ (0.0, tf), z ∈ (0.0, L)]

    @named pde_system = PDESystem([eqA, eqB, eqC, eqR], ic_bc, domains, [t, z], [CA(t, z), CB(t, z), CC(t, z)])

    dz = L / grid_size
    discretization = MOLFiniteDifference([z => dz], t, use_ODAE=true)

    prob = discretize(pde_system, discretization)

    sol = solve(prob, solver, saveat=0.2)

    # Convert the solutions to 2D arrays
    solution_array_A = Array(sol[CA(t, z)])
    solution_array_B = Array(sol[CB(t, z)])
    solution_array_C = Array(sol[CC(t, z)])

    return solution_array_A, solution_array_B, solution_array_C
end

function adv_disp_eqrxn_all(params::PyDict)
    v = params["v"]
    DaxialA = params["DaxialA"]
    DaxialB = params["DaxialB"]
    DaxialC = params["DaxialC"]
    L = params["L"]
    tf = params["tf"]
    solver_name = params["method"]
    grid_size = params["grid_size"]
    K_eq = params["equilibrium_constant"]

    solver = get_solver(solver_name)

    @parameters t, z
    @variables CA(..) CB(..) CC(..)
    Dt = Differential(t)
    Dz = Differential(z)
    Dzz = Differential(z)^2

    C_in_A(t) = sin(t) + 1.1  # Ensure it's greater than 0
    C_in_B(t) = 0.5 
    C_in_C(t) = 0

    eqA = Dt(CA(t, z)) ~ DaxialA * Dzz(CA(t, z)) - v * Dz(CA(t, z)) - K_eq * CA(t, z) * CB(t, z)
    eqB = Dt(CB(t, z)) ~ DaxialB * Dzz(CB(t, z)) - v * Dz(CB(t, z)) - K_eq * CA(t, z) * CB(t, z)
    eqC = Dt(CC(t, z)) ~ DaxialC * Dzz(CC(t, z)) - v * Dz(CC(t, z)) + K_eq * CA(t, z) * CB(t, z)
    eqR = CA(t, z) ~ K_eq * CB(t, z)

    ic_bc = [
        Dz(CA(t, 0)) ~ v * (CA(t, 0) - C_in_A(t)) / DaxialA,
        Dz(CA(t, L)) ~ 0,
        CA(0, z) ~ 1e-2,

        Dz(CB(t, 0)) ~ v * (CB(t, 0) - C_in_B(t)) / DaxialB,
        Dz(CB(t, L)) ~ 0,
        CB(0, z) ~ 1e-2,

        Dz(CC(t, 0)) ~ v * (CC(t, 0) - C_in_C(t)) / DaxialC,
        Dz(CC(t, L)) ~ 0,
        CC(0, z) ~ 1e-2
    ]

    domains = [t ∈ (0.0, tf), z ∈ (0.0, L)]

    @named pde_system = PDESystem([eqA, eqB, eqC, eqR], ic_bc, domains, [t, z], [CA(t, z), CB(t, z), CC(t, z)])

    dz = L / grid_size
    discretization = MOLFiniteDifference([z => dz], t, use_ODAE=true)

    prob = discretize(pde_system, discretization)

    sol = solve(prob, solver, saveat=0.2)

    # Convert the solutions to 2D arrays
    solution_array_A = Array(sol[CA(t, z)])
    solution_array_B = Array(sol[CB(t, z)])
    solution_array_C = Array(sol[CC(t, z)])

    return solution_array_A, solution_array_B, solution_array_C
end