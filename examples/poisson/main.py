# -*- coding: utf-8 -*-
import numpy
from numpy import pi
import pyamg
import pyfvm
from pyfvm.form_language import integrate, Subdomain, FvmProblem, \
        dS, dV, n_dot_grad
from sympy import sin


class Gamma0(Subdomain):
    def is_inside(self, x): return x[1] < 0.5
    is_boundary_only = True


class Gamma1(Subdomain):
    def is_inside(self, x): return x[1] >= 0.5
    is_boundary_only = True


class Poisson(FvmProblem):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
             - integrate(lambda x: 10 * sin(2*pi*x[0]), dV)

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - 0.0, Gamma0()),
            (lambda x: u(x) - 1.0, Gamma1())
            ]


# # Read the mesh from file
# mesh, _, _ = pyfvm.reader.read('circle.vtu')

# # Create mesh using meshzoo
# import meshzoo
# vertices, cells = meshzoo.cube.create_mesh(
#         0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
#         25, 25, 25
#         )
# mesh = pyfvm.meshTetra.meshTetra(vertices, cells)
# vertices, cells = meshzoo.rectangle.create_mesh(
#         0.0, 2.0,
#         0.0, 1.0,
#         401, 201,
#         zigzag=True
#         )
# mesh = pyfvm.meshTri.meshTri(vertices, cells)

import mshr
import dolfin
h = 2.5e-2
# cell_size = 2 * pi / num_boundary_points
c = mshr.Circle(dolfin.Point(0., 0., 0.), 1, int(2*pi / h))
# cell_size = 2 * bounding_box_radius / res
m = mshr.generate_mesh(c, 2.0 / h)
coords = m.coordinates()
coords = numpy.c_[coords, numpy.zeros(len(coords))]
mesh = pyfvm.meshTri.meshTri(coords, m.cells())

matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)

ml = pyamg.smoothed_aggregation_solver(matrix)
u = ml.solve(rhs, tol=1e-10)
# from scipy.sparse import linalg
# u = linalg.spsolve(linear_system.matrix, linear_system.rhs)

mesh.write('out.vtu', point_data={'u': u})
