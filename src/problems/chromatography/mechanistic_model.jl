using Pkg
Pkg.activate(".")
Pkg.instantiate()

using OrdinaryDiffEq
using Plots
using LinearAlgebra
pgfplotsx()

include("src/problems/chromatography/OCFEM.jl")

# OCFEM setup
n_elements = 42
collocation_points = 2
n_components = 1
n_phases = 2
p_order = 4
n_variables = n_components * n_phases * (p_order + 2 * n_elements - 2)
xₘᵢₙ, xₘₐₓ = 0.0, 1.0
h = (xₘₐₓ - xₘᵢₙ) / n_elements

H, A, B = make_OCFEM(n_elements, n_phases, n_components)
MM = BitMatrix(Array(make_MM_2(n_elements, n_phases, n_components)))

# PDE parameters
Qf, d, L = 5.0e-2, 0.5, 2.0
a, epsilon = pi*d^2/4, 0.5
u = Qf/(a*epsilon)
Pe, Dax = 21.095632695978704, u*L/Pe
cin, k_transf = 5.5, 0.22
k_iso, qmax = 1.8, 55.54

y_dy = Array(A * H^-1)
y_dy2 = Array(B * H^-1)

# Initial condition
function y_initial(y0_cache, c0)
    var0 = ones(Float64, n_variables)
    cl_idx, cu_idx = 2, p_order + 2 * n_elements - 3
    cbl_idx, cbu_idx = 1, p_order + 2 * n_elements - 2
    var0[cl_idx:cu_idx] .= c0
    var0[cbl_idx] = var0[cbu_idx] = c0
    ql_idx2, qu_idx2 = 1 * (p_order + 2 * n_elements - 2) + 1, 2 * (p_order + 2 * n_elements - 2)
    var0[ql_idx2:qu_idx2] .= qmax*k_iso*c0^1.0/(1.0 + k_iso*c0^1.0)
    var0
end

# Model definition
mutable struct col_model_node1
    n_variables; n_elements; p_order; L; h; u; y_dy; y_dy2; Pe; epsilon; c_in; dy_du; dy2_du
end

function (f::col_model_node1)(yp, y, p, t)
    dy_du, dy2_du = f.y_dy * y, f.y_dy2 * y
    c = @view y[2:f.p_order + 2*f.n_elements - 2]
    q = @view y[f.p_order + 2*f.n_elements - 1:end]
    q_eq = qmax * k_iso .* abs.(c).^1.0 ./ (1.0 .+ k_iso .* abs.(c).^1.0)
    
    yp[2:f.p_order + 2*f.n_elements - 3] .= 
        -(1 - f.epsilon) / f.epsilon * k_transf * (q_eq[2:end-1] + 0.2789*q_eq[2:end-1].*exp.(-q[2:end-1]./2.0./q_eq[2:end-1]) - q[2:end-1]) .-
        f.u * (@view dy_du[2:f.p_order + 2*f.n_elements - 3]) / f.h / f.L .+
        Dax / (f.L^2) * (@view dy2_du[2:f.p_order + 2*f.n_elements - 3]) / (f.h^2)
    
    yp[f.p_order + 2*f.n_elements - 1:end] .= 
        k_transf * (q_eq + 0.2789*q_eq.*exp.(-q./2.0./q_eq) - q)
    
    yp[1] = Dax / f.L * dy_du[1] / f.h - f.u * (y[1] - f.c_in)
    yp[f.p_order + 2*f.n_elements - 2] = dy_du[f.p_order + 2*f.n_elements - 2] / f.h / f.L
    nothing
end

# Setup and solve ODE problem
y0 = y_initial(ones(Float64, n_variables), 1e-3)
tspan = (0.0, 110.0)
rhs = col_model_node1(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, Pe, epsilon, cin, zeros(n_variables), zeros(n_variables))
f_node = ODEFunction(rhs, mass_matrix = MM)
prob_node = ODEProblem(f_node, y0, tspan, 2.0)

LinearAlgebra.BLAS.set_num_threads(1)
solution = solve(prob_node, FBDF(autodiff = false), abstol = 1e-6, reltol = 1e-6, saveat = 1.0)

# Plot results
plot(solution.t, Array(solution)[Int(n_variables/2), :], 
     label="Liquid phase concentration", 
     xlabel="Time (min)", 
     ylabel="Concentration (mg/L)",
     title="Chromatography Simulation Results")