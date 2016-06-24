# -*- coding: utf-8 -*-
#
import meshzoo
import pyfvm
from pyfvm.compiler.form_language import *
from scipy.sparse import linalg


class Singular(LinearFvmProblem):
    @staticmethod
    def apply(u):
        return integrate(lambda x: - 1.0e-2 * n_dot_grad(u(x)), dS) \
               + integrate(lambda x: u(x), dV) \
               - integrate(lambda x: 1.0, dV)
    dirichlet = [(lambda x: 0.0, ['Boundary'])]

# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle.create_mesh(
        0.0, 1.0,
        0.0, 1.0,
        51, 51,
        zigzag=True
        )
mesh = pyfvm.meshTri.meshTri(vertices, cells)

linear_system = pyfvm.discretize(Singular, mesh)

x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

mesh.write('out.vtu', point_data={'x': x})
