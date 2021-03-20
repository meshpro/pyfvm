import meshplex
import meshzoo
import numpy as np

import pyfvm
from pyfvm.form_language import Boundary, dS, dV, integrate, n_dot_grad


class Poisson:
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) - integrate(lambda x: 1.0, dV)

    def dirichlet(self, u):
        return [(lambda x: u(x) - 0.0, Boundary())]


n = 400
vertices, cells = meshzoo.rectangle_tri((0.0, 0.0), (1.0, 1.0), n)
mesh = meshplex.Mesh(vertices, cells)

matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)

# u = linalg.spsolve(matrix, rhs)
# ml = pyamg.smoothed_aggregation_solver(matrix)
# u = ml.solve(rhs, tol=1e-10)
