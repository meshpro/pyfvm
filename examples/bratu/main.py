# -*- coding: utf-8 -*-
import meshzoo
import pyfvm
from pyfvm.form_language import *
import numpy
from sympy import exp


class Bratu(FvmProblem):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
             - integrate(lambda x: 2.0 * exp(u(x)), dV)

    def dirichlet(self, u):
        return [
            (u, 'boundary')
            ]

# # Read the mesh from file
# mesh, _, _ = pyfvm.reader.read('circle.vtu')

# Create mesh using meshzoo
# vertices, cells = meshzoo.cube.create_mesh(
#         0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
#         25, 25, 25
#         )
# mesh = pyfvm.meshTetra.meshTetra(vertices, cells)
vertices, cells = meshzoo.rectangle.create_mesh(
        0.0, 2.0,
        0.0, 1.0,
        401, 201,
        zigzag=True
        )
mesh = pyfvm.meshTri.meshTri(vertices, cells)

residual, jacobian = pyfvm.discretize(Bratu(), mesh)

u0 = numpy.zeros(len(vertices))
u = pyfvm.newton(mesh, residual, jacobian, u0)

mesh.write('out.vtu', point_data={'u': u})
