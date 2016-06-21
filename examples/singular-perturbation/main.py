# -*- coding: utf-8 -*-
# ==============================================================================
from pyfvm.compiler.form_language import *


class Singular(LinearFvmProblem):
    @staticmethod
    def apply(u):
        return integrate(lambda x: - 0.01 * n_dot_grad(u(x)), dS) \
               + integrate(lambda x: u(x), dV) \
               - integrate(lambda x: 1.0, dV)
    dirichlet = [(lambda x: 0.0, Boundary)]
# ==============================================================================
import pyfvm
pyfvm.compiler.compile_classes([Singular], 'singular')

import singular

import meshzoo
from scipy.sparse import linalg

# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle.create_mesh(1.0, 1.0, 51, 51, zigzag=True)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

problem = singular.Singular(mesh)

x = linalg.spsolve(problem.matrix, problem.rhs)

mesh.write('out.vtu', point_data={'x': x})
