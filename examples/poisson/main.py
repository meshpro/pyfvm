# -*- coding: utf-8 -*-
# ==============================================================================
from pyfvm.compiler.form_language import *
from sympy import sin


class Gamma0(Subdomain):
    def is_inside(self, x): return x[1] < 0
    is_boundary_only = True


class Gamma1(Subdomain):
    def is_inside(self, x): return x[1] >= 0
    is_boundary_only = True


class Poisson(LinearFvmProblem):
    @staticmethod
    def apply(u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
                - integrate(lambda x: 10 * sin(10*x[0]), dV)

    dirichlet = [
            (lambda x: 0.0, Gamma0),
            (lambda x: 1.0, Gamma1)
            ]
# ==============================================================================
import pyfvm
pyfvm.compiler.compile_classes([Gamma0, Gamma1, Poisson], 'poisson')

import poisson

import meshzoo
from scipy.sparse import linalg

# Read the mesh using meshio
# mesh, _, _ = pyfvm.reader.read('pacman.e')

# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle.create_mesh(2.0, 1.0, 21, 11, zigzag=True)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

mesh.mark_subdomains([
        poisson.Gamma0(),
        poisson.Gamma1()
        ])

problem = poisson.Poisson(mesh)

x = linalg.spsolve(problem.matrix, problem.rhs)

mesh.write('out.vtu', point_data={'x': x})
