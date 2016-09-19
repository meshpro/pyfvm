# -*- coding: utf-8 -*-
import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dot, n, dS, dV
import meshzoo
from scipy.sparse import linalg
import sympy


class DC(object):
    def apply(self, u):
        a = sympy.Matrix([2, 1, 0])
        return \
            integrate(lambda x: -n_dot_grad(u(x)) + dot(a.T, n) * u(x), dS) \
            - integrate(lambda x: 1.0, dV)

    def dirichlet(self, u):
        return [(u, 'boundary')]


vertices, cells = meshzoo.rectangle.create_mesh(0.0, 1.0, 0.0, 1.0, 51, 51)
mesh = pyfvm.mesh_tri.MeshTri(vertices, cells)

matrix, rhs = pyfvm.discretize_linear(DC(), mesh)

u = linalg.spsolve(matrix, rhs)

mesh.write('out.vtu', point_data={'u': u})
