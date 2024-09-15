using ModelingToolkit
using OrdinaryDiffEq
using MethodOfLines, DomainSets
using Interpolations
using PythonCall

export get_solver

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