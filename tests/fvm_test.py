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

    def poisson(self, mesh, alpha, beta):
        from scipy.sparse import linalg

        # Define the problem
        class Poisson(LinearFvmProblem):
            @staticmethod
            def apply(u):
                return integrate(lambda x: - n_dot_grad(u(x)), dS) \
                       - integrate(lambda x: 1.0, dV)
            dirichlet = [(lambda x: 0.0, ['Boundary'])]

        linear_system = pyfvm.discretize(Poisson, mesh)

        x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

        k0 = -1
        for k, coord in enumerate(mesh.node_coords):
            # print(coord - [0.5, 0.5, 0.0])
            if numpy.linalg.norm(coord - [0.5, 0.5, 0.0]) < 1.0e-5:
                k0 = k
                break

        self.assertNotEqual(k0, -1)
        self.assertAlmostEqual(x[k0], alpha, delta=1.0e-7)

        x_dot_x = numpy.dot(x, mesh.control_volumes * x)
        self.assertAlmostEqual(x_dot_x, beta, delta=1.0e-7)

        return

    def test_poisson_2d(self):
        import meshzoo
        vertices, cells = meshzoo.rectangle.create_mesh(
                0.0, 1.0, 0.0, 1.0,
                21, 21,
                zigzag=True
                )
        mesh = pyfvm.meshTri.meshTri(vertices, cells)
        self.poisson(
                mesh,
                0.0735267092334,
                0.001695424171463697
                )
        return

    def test_poisson_3d(self):
        import meshzoo
        vertices, cells = meshzoo.cube.create_mesh(
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                11, 11, 11
                )
        mesh = pyfvm.meshTetra.meshTetra(vertices, cells)
        self.poisson(
                mesh,
                0.0,
                0.00061344097536402699
                )
        return

    def test_boundaries(self):
        import meshzoo
        from scipy.sparse import linalg

        class Gamma0(Subdomain):
            def is_inside(self, x): return x[1] < 0.5
            is_boundary_only = True

        class Gamma1(Subdomain):
            def is_inside(self, x): return x[1] >= 0.5
            is_boundary_only = True

        # Define the problem
        class Poisson(LinearFvmProblem):
            @staticmethod
            def apply(u):
                return integrate(lambda x: -n_dot_grad(u(x)), dS) \
                       - integrate(lambda x: 1.0, dV)
            dirichlet = [
                    (lambda x: 0.0, ['Gamma0']),
                    (lambda x: 1.0, ['Gamma1'])
                    ]

        # Create mesh using meshzoo
        vertices, cells = meshzoo.rectangle.create_mesh(
                0.0, 1.0, 0.0, 1.0,
                21, 21,
                zigzag=True
                )
        mesh = pyfvm.meshTri.meshTri(vertices, cells)
        mesh.mark_subdomains([Gamma0(), Gamma1()])

        linear_system = pyfvm.discretize(Poisson, mesh)

        x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

        k0 = -1
        for k, coord in enumerate(mesh.node_coords):
            # print(coord - [0.5, 0.5, 0.0])
            if numpy.linalg.norm(coord - [0.5, 0.5, 0.0]) < 1.0e-5:
                k0 = k
                break

        self.assertNotEqual(k0, -1)
        self.assertAlmostEqual(x[k0], 0.59455184740329481, delta=1.0e-7)

        x_dot_x = numpy.dot(x, mesh.control_volumes * x)
        self.assertAlmostEqual(x_dot_x, 0.42881926935620163, delta=1.0e-7)

        return

    def test_singular_perturbation(self):
        import meshzoo
        from scipy.sparse import linalg

        # Define the problem
        class Poisson(LinearFvmProblem):
            @staticmethod
            def apply(u):
                return integrate(lambda x: - 1.0e-2 * n_dot_grad(u(x)), dS) \
                       + integrate(lambda x: u(x), dV) \
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

        self.assertNotEqual(k0, -1)
        self.assertAlmostEqual(x[k0], 0.97335485230869123, delta=1.0e-7)

        x_dot_x = numpy.dot(x, mesh.control_volumes * x)
        self.assertAlmostEqual(x_dot_x, 0.49724636865618776, delta=1.0e-7)

        return

    def test_neumann(self):
        import meshzoo
        from scipy.sparse import linalg

        class D1(Subdomain):
            def is_inside(self, x): return x[1] < 0.5
            is_boundary_only = True

        # Define the problem
        class Poisson(LinearFvmProblem):
            @staticmethod
            def apply(u):
                return integrate(lambda x: - n_dot_grad(u(x)), dS) \
                       + integrate(lambda x: 3.0, dGamma) \
                       - integrate(lambda x: 1.0, dV)
            dirichlet = [(lambda x: 0.0, ['D1'])]

        # Create mesh using meshzoo
        vertices, cells = meshzoo.rectangle.create_mesh(
                0.0, 1.0, 0.0, 1.0,
                21, 21,
                zigzag=True
                )
        mesh = pyfvm.meshTri.meshTri(vertices, cells)
        mesh.mark_subdomains([D1()])

        linear_system = pyfvm.discretize(Poisson, mesh)

        x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

        k0 = -1
        for k, coord in enumerate(mesh.node_coords):
            # print(coord - [0.5, 0.5, 0.0])
            if numpy.linalg.norm(coord - [0.5, 0.5, 0.0]) < 1.0e-5:
                k0 = k
                break

        self.assertNotEqual(k0, -1)
        self.assertAlmostEqual(x[k0], -1.3249459366260112, delta=1.0e-7)

        x_dot_x = numpy.dot(x, mesh.control_volumes * x)
        self.assertAlmostEqual(x_dot_x, 3.1844205150779601, delta=1.0e-7)

        return

    # TODO reinstate test
    # def test_convection(self):
    #     import meshzoo
    #     from scipy.sparse import linalg

    #     # Define the problem
    #     class Poisson(LinearFvmProblem):
    #         @staticmethod
    #         def apply(u):
    #             a = sympy.Matrix([2, 1, 0])
    #             return integrate(lambda x: - n_dot_grad(u(x)) + dot(n, a) * u(x), dS) - \
    #                    integrate(lambda x: 1.0, dV)
    #         dirichlet = [(lambda x: 0.0, ['Boundary'])]

    #     # Create mesh using meshzoo
    #     vertices, cells = meshzoo.rectangle.create_mesh(
    #             0.0, 1.0, 0.0, 1.0,
    #             21, 21,
    #             zigzag=True
    #             )
    #     mesh = pyfvm.meshTri.meshTri(vertices, cells)

    #     linear_system = pyfvm.discretize(Poisson, mesh)

    #     x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

    #     k0 = -1
    #     for k, coord in enumerate(mesh.node_coords):
    #         if numpy.linalg.norm(coord - [0.5, 0.5, 0.0]) < 1.0e-5:
    #             k0 = k
    #             break

    #     self.assertNotEqual(k0, -1)
    #     self.assertAlmostEqual(x[k0], 0.07041709172659899, delta=1.0e-7)

    #     x_dot_x = numpy.dot(x, mesh.control_volumes * x)
    #     self.assertAlmostEqual(x_dot_x, 0.0, delta=1.0e-7)

    #     return


if __name__ == '__main__':
    unittest.main()
