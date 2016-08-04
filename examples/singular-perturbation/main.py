# -*- coding: utf-8 -*-
#
import meshzoo
import pyfvm
from pyfvm.form_language import *
from scipy.sparse import linalg


class Singular(LinearFvmProblem):
    def apply(self, u):
        return integrate(lambda x: - 1.0e-2 * n_dot_grad(u(x)), dS) \
               + integrate(u, dV) \
               - integrate(lambda x: 1.0, dV)

    def dirichlet(self, u):
        return [(u, 'boundary')]

# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle.create_mesh(
        0.0, 1.0,
        0.0, 1.0,
        51, 51,
        zigzag=True
        )
mesh = pyfvm.meshTri.meshTri(vertices, cells)

linear_system = pyfvm.discretize(Singular(), mesh)

x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

mesh.write('out.vtu', point_data={'x': x})
