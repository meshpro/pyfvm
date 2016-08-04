# -*- coding: utf-8 -*-
import pyfvm
from pyfvm.form_language import *
import meshzoo
from scipy.sparse import linalg


class DC(LinearFvmProblem):
    def apply(self, u):
        a = sympy.Matrix([2, 1, 0])
        return \
            integrate(lambda x: -n_dot_grad(u(x)) + dot(a.T, n) * u(x), dS) - \
            integrate(lambda x: 1.0, dV)

    def dirichlet(self, u):
        return [(u, 'boundary')]


vertices, cells = meshzoo.rectangle.create_mesh(
        0.0, 1.0,
        0.0, 1.0,
        51, 51,
        zigzag=True
        )
mesh = pyfvm.meshTri.meshTri(vertices, cells)

linear_system = pyfvm.discretize(DC(), mesh)

x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

mesh.write('out.vtu', point_data={'x': x})
