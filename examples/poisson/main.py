# -*- coding: utf-8 -*-
import numpy
from numpy import pi
import pyamg
import pyfvm
from pyfvm.form_language import integrate, Subdomain, FvmProblem, \
        dS, dV, n_dot_grad
from sympy import sin
import voropy


class Gamma0(Subdomain):
    def is_inside(self, x): return x[1] < 0.5
    is_boundary_only = True


class Gamma1(Subdomain):
    def is_inside(self, x): return x[1] >= 0.5
    is_boundary_only = True


class Poisson(FvmProblem):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
             - integrate(lambda x: 50 * sin(2*pi*x[0]), dV)

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - 0.0, Gamma0()),
            (lambda x: u(x) - 1.0, Gamma1())
            ]


# # Read the mesh from file
# mesh, _, _ = pyfvm.reader.read('circle.vtu')

# Create mesh using meshzoo
import meshzoo
vertices, cells = meshzoo.cube(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 50, 50, 50)
mesh = voropy.mesh_tetra.MeshTetra(vertices, cells)
# vertices, cells = meshzoo.rectangle(0.0, 2.0, 0.0, 1.0, 401, 201)
# mesh = voropy.mesh_tri.MeshTri(vertices, cells)
print(len(vertices))

# import mshr
# import dolfin
# # h = 2.5e-3
# h = 1.e-1
# # cell_size = 2 * pi / num_boundary_points
# c = mshr.Circle(dolfin.Point(0., 0., 0.), 1, int(2*pi / h))
# # cell_size = 2 * bounding_box_radius / res
# m = mshr.generate_mesh(c, 2.0 / h)
# coords = m.coordinates()
# coords = numpy.c_[coords, numpy.zeros(len(coords))]
# cells = m.cells().copy()
# mesh = voropy.mesh_tri.MeshTri(coords, cells)
# # mesh = voropy.mesh_tri.lloyd_smoothing(mesh, 1.0e-4)

matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)

#ml = pyamg.smoothed_aggregation_solver(matrix)
ml = pyamg.ruge_stuben_solver(matrix)
u = ml.solve(rhs, tol=1e-10)
# from scipy.sparse import linalg
# u = linalg.spsolve(matrix, rhs)

# mesh.write('out.vtu', point_data={'u': u})
