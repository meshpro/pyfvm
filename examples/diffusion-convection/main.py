# -*- coding: utf-8 -*-
import pyfvm
from pyfvm.form_language import *
import meshzoo
from scipy.sparse import linalg


class DC(LinearFvmProblem):
    @staticmethod
    def apply(u):
        a = sympy.Matrix([0, 1, 0])
        return \
            integrate(lambda x: -n_dot_grad(u(x)) + dot(n, a) * u(x), dS) - \
            integrate(lambda x: 1.0, dV)

    dirichlet = [(lambda x: 0.0, ['Boundary'])]


vertices, cells = meshzoo.rectangle.create_mesh(
        0.0, 1.0,
        0.0, 1.0,
        51, 51,
        zigzag=True
        )
mesh = pyfvm.meshTri.meshTri(vertices, cells)

linear_system = pyfvm.discretize(DC, mesh)

x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

mesh.write('out.vtu', point_data={'x': x})
