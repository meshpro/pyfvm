# -*- coding: utf-8 -*-
import pyamg
import pyfvm
import pygmsh
from pyfvm.form_language import *
import meshzoo
from scipy.sparse import linalg
from sympy import sin
import numpy
from numpy import pi
import mshr
import dolfin


class Gamma0(Subdomain):
    def is_inside(self, x): return x[1] < 0.5
    is_boundary_only = True


class Gamma1(Subdomain):
    def is_inside(self, x): return x[1] >= 0.5
    is_boundary_only = True


class Poisson(LinearFvmProblem):
    @staticmethod
    def apply(u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
                - integrate(lambda x: 10 * sin(2*pi*x[0]), dV)

    dirichlet = [
            (lambda x: 0.0, ['Gamma0']),
            (lambda x: 1.0, ['Gamma1'])
            # (lambda x: 0.0, ['Boundary'])
            ]

# # Read the mesh using meshio
# mesh, _, _ = pyfvm.reader.read('circle.vtu')

# # Create mesh using meshzoo
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


h = 2.5e-3
# cell_size = 2 * pi / num_boundary_points
c = mshr.Circle(dolfin.Point(0., 0., 0.), 1, int(2*pi / h))
# cell_size = 2 * bounding_box_radius / res
m = mshr.generate_mesh(c, 2.0 / h)
coords = m.coordinates()
coords = numpy.c_[coords, numpy.zeros(len(coords))]
print(len(coords))
mesh = pyfvm.meshTri.meshTri(coords, m.cells())

mesh.mark_subdomains([Gamma0(), Gamma1()])

linear_system = pyfvm.discretize(Poisson, mesh)

ml = pyamg.ruge_stuben_solver(linear_system.matrix)
x = ml.solve(linear_system.rhs, tol=1e-10)
# x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

mesh.write('out.vtu', point_data={'x': x})
