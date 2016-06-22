# -*- coding: utf-8 -*-
#
import os
import numpy
import unittest

import pyfvm
from pyfvm.compiler.form_language import *


class TestControlVolumes(unittest.TestCase):

    def setUp(self):
        return

    def test_poisson(self):
        class Poisson(LinearFvmProblem):
            @staticmethod
            def apply(u):
                return integrate(lambda x: - n_dot_grad(u(x)), dS) \
                       - integrate(lambda x: 1.0, dV)
            dirichlet = [(lambda x: 1.0, Boundary)]
        # ======================================================================
        import pyfvm
        pyfvm.compiler.compile_classes([Poisson], 'poisson_def')

        import poisson_def

        import meshzoo
        from scipy.sparse import linalg

        # Create mesh using meshzoo
        vertices, cells = meshzoo.rectangle.create_mesh(
                1.0, 1.0,
                21, 21,
                zigzag=True
                )
        mesh = pyfvm.meshTri.meshTri(vertices, cells)

        problem = poisson_def.Poisson(mesh)

        x = linalg.spsolve(problem.matrix, problem.rhs)

        k0 = -1
        for k, coord in enumerate(mesh.node_coords):
            # print(coord - [0.5, 0.5, 0.0])
            if numpy.linalg.norm(coord - [0.5, 0.5, 0.0]) < 1.0e-5:
                k0 = k
                break

        self.assertAlmostEqual(x[k0], 1.0, delta=1.0e-7)

        return


if __name__ == '__main__':
    unittest.main()
