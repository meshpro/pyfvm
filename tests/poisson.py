# -*- coding: utf-8 -*-
#
import os
import numpy
import unittest

import pyfvm
from pyfvm.form_language import *


class TestPDEs(unittest.TestCase):

    def setUp(self):
        return

    def test_poisson(self):
        import meshzoo
        from scipy.sparse import linalg

        # Define the problem
        class Poisson(LinearFvmProblem):
            @staticmethod
            def apply(u):
                return integrate(lambda x: - n_dot_grad(u(x)), dS) \
                       - integrate(lambda x: 1.0, dV)
            dirichlet = [(lambda x: 0.0, ['Boundary'])]

        # Create mesh using meshzoo
        vertices, cells = meshzoo.rectangle.create_mesh(
                0.0, 1.0, 0.0, 1.0,
                21, 21,
                zigzag=True
                )
        mesh = pyfvm.meshTri.meshTri(vertices, cells)

        linear_system = pyfvm.discretize(Poisson, mesh)

        x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

        k0 = -1
        for k, coord in enumerate(mesh.node_coords):
            # print(coord - [0.5, 0.5, 0.0])
            if numpy.linalg.norm(coord - [0.5, 0.5, 0.0]) < 1.0e-5:
                k0 = k
                break

        # self.assertNotEqual(k0, -1)
        self.assertAlmostEqual(x[k0], 0.0735267092334, delta=1.0e-7)

        return


if __name__ == '__main__':
    unittest.main()
