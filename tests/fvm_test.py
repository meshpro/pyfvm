# -*- coding: utf-8 -*-
#
import numpy
import sympy
import unittest

import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, \
        dS, dV, dGamma, Subdomain, dot, n


class TestPDEs(unittest.TestCase):

    def setUp(self):
        return

    def poisson(self, mesh, alpha, beta, gamma):
        from scipy.sparse import linalg

        # Define the problem
        class Poisson(object):
            def apply(self, u):
                return integrate(lambda x: - n_dot_grad(u(x)), dS) \
                       - integrate(lambda x: 1.0, dV)

            def dirichlet(self, u):
                return [(u, 'boundary')]

        matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)

        u = linalg.spsolve(matrix, rhs)

        norm1 = numpy.sum(mesh.control_volumes * abs(u))
        self.assertAlmostEqual(norm1, alpha, delta=1.0e-7)

        norm2 = numpy.sqrt(numpy.sum(mesh.control_volumes * abs(u)**2))
        self.assertAlmostEqual(norm2, beta, delta=1.0e-7)

        norm_inf = numpy.max(abs(u))
        self.assertAlmostEqual(norm_inf, gamma, delta=1.0e-7)

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
                0.03486068399064314,
                0.041175528793977852,
                0.073526709233390164
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
                0.019193629907227189,
                0.02476774061887816,
                0.055797992672053209
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
        class Poisson(object):
            def apply(self, u):
                return integrate(lambda x: -n_dot_grad(u(x)), dS) \
                       - integrate(lambda x: 1.0, dV)

            def dirichlet(self, u):
                return [
                    (lambda x: u(x) - 0.0, Gamma0()),
                    (lambda x: u(x) - 1.0, Gamma1())
                    ]

        # Create mesh using meshzoo
        vertices, cells = meshzoo.rectangle.create_mesh(
                0.0, 1.0, 0.0, 1.0,
                21, 21,
                zigzag=True
                )
        mesh = pyfvm.meshTri.meshTri(vertices, cells)

        matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)

        u = linalg.spsolve(matrix, rhs)

        norm1 = numpy.sum(mesh.control_volumes * abs(u))
        self.assertAlmostEqual(norm1, 0.55175078324384885, delta=1.0e-7)

        norm2 = numpy.sqrt(numpy.sum(mesh.control_volumes * abs(u)**2))
        self.assertAlmostEqual(norm2, 0.65484293487538081, delta=1.0e-7)

        norm_inf = numpy.max(abs(u))
        self.assertAlmostEqual(norm_inf, 1.0, delta=1.0e-7)

        return

    def test_singular_perturbation(self):
        import meshzoo
        from scipy.sparse import linalg

        # Define the problem
        class Poisson(object):
            def apply(self, u):
                return integrate(lambda x: - 1.0e-2 * n_dot_grad(u(x)), dS) \
                       + integrate(lambda x: u(x), dV) \
                       - integrate(lambda x: 1.0, dV)

            def dirichlet(self, u):
                return [(u, 'boundary')]

        # Create mesh using meshzoo
        vertices, cells = meshzoo.rectangle.create_mesh(
                0.0, 1.0, 0.0, 1.0,
                21, 21,
                zigzag=True
                )
        mesh = pyfvm.meshTri.meshTri(vertices, cells)

        matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)

        u = linalg.spsolve(matrix, rhs)

        norm1 = numpy.sum(mesh.control_volumes * abs(u))
        self.assertAlmostEqual(norm1, 0.64033452662531842, delta=1.0e-7)

        norm2 = numpy.sqrt(numpy.sum(mesh.control_volumes * abs(u)**2))
        self.assertAlmostEqual(norm2, 0.70515698156948536, delta=1.0e-7)

        norm_inf = numpy.max(abs(u))
        self.assertAlmostEqual(norm_inf, 0.97335485230869012, delta=1.0e-7)

        return

    def test_neumann(self):
        import meshzoo
        from scipy.sparse import linalg

        class D1(Subdomain):
            def is_inside(self, x): return x[1] < 0.5
            is_boundary_only = True

        # Define the problem
        class Poisson(object):
            def apply(self, u):
                return integrate(lambda x: - n_dot_grad(u(x)), dS) \
                       + integrate(lambda x: 3.0, dGamma) \
                       - integrate(lambda x: 1.0, dV)

            def dirichlet(self, u):
                return [(u, D1())]

        # Create mesh using meshzoo
        vertices, cells = meshzoo.rectangle.create_mesh(
                0.0, 1.0, 0.0, 1.0,
                21, 21,
                zigzag=True
                )
        mesh = pyfvm.meshTri.meshTri(vertices, cells)

        matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)

        u = linalg.spsolve(matrix, rhs)

        norm1 = numpy.sum(mesh.control_volumes * abs(u))
        self.assertAlmostEqual(norm1, 1.4064456286297495, delta=1.0e-7)

        norm2 = numpy.sqrt(numpy.sum(mesh.control_volumes * abs(u)**2))
        self.assertAlmostEqual(norm2, 1.7844944704531931, delta=1.0e-7)

        norm_inf = numpy.max(abs(u))
        self.assertAlmostEqual(norm_inf, 3.7887143197122652, delta=1.0e-7)

        return

    def test_convection(self):
        import meshzoo
        from scipy.sparse import linalg

        # Define the problem
        class Poisson(object):
            def apply(self, u):
                a = sympy.Matrix([2, 1, 0])
                return integrate(
                        lambda x: - n_dot_grad(u(x)) + dot(a.T, n) * u(x), dS
                        ) \
                    - integrate(lambda x: 1.0, dV)

            def dirichlet(self, u):
                return [(u, 'boundary')]

        # Create mesh using meshzoo
        vertices, cells = meshzoo.rectangle.create_mesh(
                0.0, 1.0, 0.0, 1.0,
                21, 21,
                zigzag=True
                )
        mesh = pyfvm.meshTri.meshTri(vertices, cells)

        matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)

        u = linalg.spsolve(matrix, rhs)

        norm1 = numpy.sum(mesh.control_volumes * abs(u))
        self.assertAlmostEqual(norm1, 0.033893251703668408, delta=1.0e-7)

        norm2 = numpy.sqrt(numpy.sum(mesh.control_volumes * abs(u)**2))
        self.assertAlmostEqual(norm2, 0.040095425713737309, delta=1.0e-7)

        norm_inf = numpy.max(abs(u))
        self.assertAlmostEqual(norm_inf, 0.071790294919364853, delta=1.0e-7)

        return

    def test_bratu(self):
        import meshzoo
        from sympy import exp

        class Bratu(object):
            def apply(self, u):
                return integrate(lambda x: - n_dot_grad(u(x)), dS) \
                       - integrate(lambda x: 2.0 * exp(u(x)), dV)

            def dirichlet(self, u):
                return [(u, 'boundary')]

        # Create mesh using meshzoo
        vertices, cells = meshzoo.rectangle.create_mesh(
                0.0, 1.0, 0.0, 1.0,
                21, 21,
                zigzag=True
                )
        mesh = pyfvm.meshTri.meshTri(vertices, cells)

        f, jacobian = pyfvm.discretize(Bratu(), mesh)

        u0 = numpy.zeros(len(vertices))
        u = pyfvm.newton(f.eval, jacobian.get_linear_operator, u0, verbose=False)

        norm1 = numpy.sum(mesh.control_volumes * abs(u))
        self.assertAlmostEqual(norm1, 0.077809948662596773, delta=1.0e-7)

        norm2 = numpy.sqrt(numpy.sum(mesh.control_volumes * abs(u)**2))
        self.assertAlmostEqual(norm2, 0.092270491605142432, delta=1.0e-7)

        norm_inf = numpy.max(abs(u))
        self.assertAlmostEqual(norm_inf, 0.16660466836481022, delta=1.0e-7)

        return


if __name__ == '__main__':
    unittest.main()
