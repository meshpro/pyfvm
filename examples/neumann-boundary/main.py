# -*- coding: utf-8 -*-
import pyfvm
from pyfvm.form_language import *
import meshzoo
from scipy.sparse import linalg


class D1(Subdomain):
    def is_inside(self, x): return x[1] < 0.5
    is_boundary_only = True


class Poisson(LinearFvmProblem):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
                + integrate(lambda x: 3.0, dGamma) \
                - integrate(lambda x: 1.0, dV)

    def dirichlet(self, u):
        return [(u, D1())]

vertices, cells = meshzoo.rectangle.create_mesh(
        0.0, 1.0,
        0.0, 1.0,
        51, 51,
        zigzag=True
        )
mesh = pyfvm.meshTri.meshTri(vertices, cells)

linear_system = pyfvm.discretize(Poisson(), mesh)

x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

mesh.write('out.vtu', point_data={'x': x})
