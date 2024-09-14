from juliacall import Main as jl


jl.include("src/julia/test_diffeq.jl")

# Call functions from the Julia scripts
solution = jl.solve_simple_ode(20)
print(f"Solution: {solution}")