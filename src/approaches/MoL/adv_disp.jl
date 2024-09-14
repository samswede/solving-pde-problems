using ModelingToolkit
using OrdinaryDiffEq
using MethodOfLines, DomainSets


function solve_adv_diff(
    v = 0.05,
    C_in = 1.0,
    Daxial = 1e-3,
    L = 1.0,
    tf = 10.0
)

    @parameters t, z
    @variables C(..)
    Dt = Differential(t)
    Dz = Differential(z)
    Dzz = Differential(z)^2

    eq = Dt(C(t, z)) ~ Daxial*Dzz(C(t, z)) - v*Dz(C(t, z))

    ic_bc = [
        Dz(C(t, 0)) ~ v * (C(t, 0) - C_in) / Daxial,
        Dz(C(t, L)) ~ 0,
        C(0, z) ~ 1e-2
    ]

    domains = [t ∈ (0.0, tf), z ∈ (0.0, L)]

    @named pde_system = PDESystem(eq, ic_bc, domains, [t, z], [C(t, z)])

    dz = 0.01
    discretization = MOLFiniteDifference([z => dz], t)

    prob = discretize(pde_system, discretization)

    sol = solve(prob, Tsit5(), saveat=0.2)


    # Convert the solution to a simple array
    # discrete_z = sol[z]
    # discrete_t = sol[t]
    # println("Discrete z values: ", discrete_z)
    # println("Discrete t values: ", discrete_t)

    # solution_array = [sol[C(ti, zi)] for ti in discrete_t, zi in discrete_z]

    # Convert the solution to a 2D array
    solution_array = Array(sol[C(t, z)])

    return solution_array
end


function solve_adv_diff_2(
    v = 0.05,
    Daxial = 1e-3,
    L = 1.0,
    tf = 10.0
)

    @parameters t, z
    @variables C(..)
    Dt = Differential(t)
    Dz = Differential(z)
    Dzz = Differential(z)^2

    C_in(t) = sin(t) + 1

    eq = Dt(C(t, z)) ~ Daxial*Dzz(C(t, z)) - v*Dz(C(t, z))

    ic_bc = [
        Dz(C(t, 0)) ~ v * (C(t, 0) - C_in(t)) / Daxial,
        Dz(C(t, L)) ~ 0,
        C(0, z) ~ 1e-2
    ]

    domains = [t ∈ (0.0, tf), z ∈ (0.0, L)]

    @named pde_system = PDESystem(eq, ic_bc, domains, [t, z], [C(t, z)])

    dz = 0.01
    discretization = MOLFiniteDifference([z => dz], t)

    prob = discretize(pde_system, discretization)

    sol = solve(prob, Tsit5(), saveat=0.2)

    # Convert the solution to a 2D array
    solution_array = Array(sol[C(t, z)])

    return solution_array
end


function solve_adv_diff_3(
    v = 0.05,
    DaxialA = 1e-2,
    DaxialB = 1e-4,
    DaxialC = 1e-6,
    L = 1.0,
    tf = 10.0
)

    @parameters t, z
    @variables CA(..) CB(..) CC(..)
    Dt = Differential(t)
    Dz = Differential(z)
    Dzz = Differential(z)^2

    C_in_A(t) = sin(t) + 1.1  # Ensure it's greater than 0
    C_in_B(t) = cos(t) + 1.1  # Ensure it's greater than 0
    C_in_C(t) = 0.5*sin(t) + 1.1  # Ensure it's greater than 0

    eqA = Dt(CA(t, z)) ~ DaxialA*Dzz(CA(t, z)) - v*Dz(CA(t, z))
    eqB = Dt(CB(t, z)) ~ DaxialB*Dzz(CB(t, z)) - v*Dz(CB(t, z))
    eqC = Dt(CC(t, z)) ~ DaxialC*Dzz(CC(t, z)) - v*Dz(CC(t, z))

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

    @named pde_system = PDESystem([eqA, eqB, eqC], ic_bc, domains, [t, z], [CA(t, z), CB(t, z), CC(t, z)])

    dz = 0.01
    discretization = MOLFiniteDifference([z => dz], t)

    prob = discretize(pde_system, discretization)

    sol = solve(prob, Tsit5(), saveat=0.2)

    # Convert the solutions to 2D arrays
    solution_array_A = Array(sol[CA(t, z)])
    solution_array_B = Array(sol[CB(t, z)])
    solution_array_C = Array(sol[CC(t, z)])

    return solution_array_A, solution_array_B, solution_array_C
end