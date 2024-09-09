from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from dolfinx import mesh, fem, io, plot
from dolfinx.fem import (FunctionSpace, Function, Constant)
import ufl  # Import ufl for TestFunction, TrialFunction, etc.

# Time-stepping parameters
T = 5.0
num_steps = 500
dt = T / num_steps
eps = 0.01
K = 10.0

# Load mesh from file
with io.XDMFFile(MPI.COMM_WORLD, 'navier_stokes_cylinder/cylinder.xdmf', 'r') as xdmf:
    mesh = xdmf.read_mesh()

# Define function spaces for velocity and concentrations
W = fem.VectorFunctionSpace(mesh, ("Lagrange", 2))
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = ufl.MixedElement([P1, P1, P1])
V = fem.FunctionSpace(mesh, element)

# Define test and trial functions
v_1, v_2, v_3 = ufl.TestFunctions(V)
u = fem.Function(V)
u_n = fem.Function(V)
u_1, u_2, u_3 = ufl.split(u)
u_n1, u_n2, u_n3 = ufl.split(u_n)

# Define source terms and coefficients
f_1 = fem.Constant(mesh, PETSc.ScalarType(0.1))
f_2 = fem.Constant(mesh, PETSc.ScalarType(0.1))
f_3 = fem.Constant(mesh, PETSc.ScalarType(0))
k = fem.Constant(mesh, PETSc.ScalarType(dt))
K = fem.Constant(mesh, PETSc.ScalarType(K))
eps = fem.Constant(mesh, PETSc.ScalarType(eps))

# Define the variational form
F = ((u_1 - u_n1) / k) * v_1 * ufl.dx + ufl.dot(fem.Function(W), ufl.grad(u_1)) * v_1 * ufl.dx \
    + eps * ufl.dot(ufl.grad(u_1), ufl.grad(v_1)) * ufl.dx + K * u_1 * u_2 * v_1 * ufl.dx \
    + ((u_2 - u_n2) / k) * v_2 * ufl.dx + ufl.dot(fem.Function(W), ufl.grad(u_2)) * v_2 * ufl.dx \
    + eps * ufl.dot(ufl.grad(u_2), ufl.grad(v_2)) * ufl.dx + K * u_1 * u_2 * v_2 * ufl.dx \
    + ((u_3 - u_n3) / k) * v_3 * ufl.dx + ufl.dot(fem.Function(W), ufl.grad(u_3)) * v_3 * ufl.dx \
    + eps * ufl.dot(ufl.grad(u_3), ufl.grad(v_3)) * ufl.dx - K * u_1 * u_2 * v_3 * ufl.dx + K * u_3 * v_3 * ufl.dx \
    - f_1 * v_1 * ufl.dx - f_2 * v_2 * ufl.dx - f_3 * v_3 * ufl.dx

# Time-stepping
with io.XDMFFile(MPI.COMM_WORLD, 'reaction_system/u_1.xdmf', 'w') as vtkfile_u_1, \
     io.XDMFFile(MPI.COMM_WORLD, 'reaction_system/u_2.xdmf', 'w') as vtkfile_u_2, \
     io.XDMFFile(MPI.COMM_WORLD, 'reaction_system/u_3.xdmf', 'w') as vtkfile_u_3:

    t = 0
    for n in range(num_steps):
        t += dt
        
        # Solve the system
        problem = fem.petsc.NonlinearProblem(F, u)
        solver = fem.petsc.NewtonSolver(MPI.COMM_WORLD)
        solver.solve(problem, u.vector)
        
        # Write results
        _u_1, _u_2, _u_3 = u.split()
        vtkfile_u_1.write_function(_u_1, t)
        vtkfile_u_2.write_function(_u_2, t)
        vtkfile_u_3.write_function(_u_3, t)
        
        # Update previous solution
        u_n.x.array[:] = u.x.array[:]