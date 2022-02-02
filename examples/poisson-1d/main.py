import numpy as np
from scipy.sparse import linalg

import pyfvm
from pyfvm.form_language import LinearFvmProblem, dS, dV, integrate, n_dot_grad


class Poisson(LinearFvmProblem):
    @staticmethod
    def apply(u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) - integrate(lambda x: 1.0, dV)

    dirichlet = [(lambda x: 0.0, ["Boundary"])]


x = np.linspace(0.0, 1.0, 100)
cells = [[i, i + 1] for i in range(0, len(x) - 1)]
mesh = pyfvm.meshLine.meshLine(x, cells)

linear_system = pyfvm.discretize(Poisson, mesh)

x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

mesh.write("out.vtk", point_data={"x": x})
