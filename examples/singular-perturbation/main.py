# -*- coding: utf-8 -*-
#
import meshzoo
import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, dV
from scipy.sparse import linalg


class Singular(object):
    def apply(self, u):
        return integrate(lambda x: - 1.0e-2 * n_dot_grad(u(x)), dS) \
               + integrate(u, dV) \
               - integrate(lambda x: 1.0, dV)

    def dirichlet(self, u):
        return [(u, 'boundary')]

vertices, cells = meshzoo.rectangle.create_mesh(0.0, 1.0, 0.0, 1.0, 51, 51)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

matrix, rhs = pyfvm.discretize_linear(Singular(), mesh)

u = linalg.spsolve(matrix, rhs)

mesh.write('out.vtu', point_data={'u': u})
