# -*- coding: utf-8 -*-
import mshr
import dolfin
import helpers
import pyamg
import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, dV, dot, n
from sympy import pi, sin, cos, Matrix
import unittest


def exact_sol(x):
    return cos(pi/2 * (x[0]**2 + x[1]**2 + x[2]**2))


class Convection(object):
    def apply(self, u):
        a0 = 2
        a1 = 1
        a2 = 3
        a = Matrix([a0, a1, a2])

        def rhs(x):
            z = pi/2 * (x[0]**2 + x[1]**2 + x[2]**2)
            return 2*pi * (1.5 * sin(z) + z * cos(z)) - \
                a0 * pi * x[0] * sin(z) - \
                a1 * pi * x[1] * sin(z) - \
                a2 * pi * x[2] * sin(z)

        return integrate(lambda x: -n_dot_grad(u(x)) + dot(a.T, n)*u(x), dS) \
            - integrate(rhs, dV)

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - exact_sol(x), 'boundary')
            ]


def get_mesh(k):
    h = 0.5**(k+2)
    c = mshr.Sphere(dolfin.Point(0., 0., 0.), 1.0, int(2*pi / h))
    m = mshr.generate_mesh(c, 2.0 / h)
    return pyfvm.mesh_tetra.MeshTetra(
            m.coordinates(),
            m.cells(),
            mode='geometric'
            )


class ConvergenceConvection3dBallTest(unittest.TestCase):

    def setUp(self):
        return

    @staticmethod
    def solve(verbose=False):
        def solver(mesh):
            matrix, rhs = pyfvm.discretize_linear(Convection(), mesh)
            ml = pyamg.smoothed_aggregation_solver(matrix)
            u = ml.solve(rhs, tol=1e-10)
            return u

        return helpers.perform_convergence_tests(
            solver,
            exact_sol,
            get_mesh,
            range(3),
            verbose=verbose
            )

    def test(self):
        H, error_norm_1, error_norm_inf, order_1, order_inf = self.solve()

        expected_order = 2
        tol = 0.2
        self.assertGreater(order_1[-1], expected_order - tol)
        self.assertGreater(order_inf[-1], expected_order - tol)

        return


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    H, error_norm_1, error_norm_inf, order_1, order_inf = \
        ConvergenceConvection3dBallTest.solve(verbose=True)

    helpers.plot_error_data(H, error_norm_1, error_norm_inf)
    plt.show()
