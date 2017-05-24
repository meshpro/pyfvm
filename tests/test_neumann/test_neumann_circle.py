# -*- coding: utf-8 -*-
import helpers
import pyamg
import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, \
        dS, dGamma, dV, Subdomain
from sympy import pi, sin, cos, sqrt
import unittest


def exact_sol(x):
    return cos(pi/2 * (x[0]**2 + x[1]**2))


class Gamma1(Subdomain):
    def is_inside(self, x):
        return x[1] < 0.0
    is_boundary_only = True


class Neumann(object):
    def apply(self, u):
        def neumann(x):
            z = x[0]**2 + x[1]**2
            return -pi * sqrt(z) * sin(pi/2 * z)

        def rhs(x):
            z = pi/2 * (x[0]**2 + x[1]**2)
            return 2*pi * (sin(z) + z * cos(z))

        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
            - integrate(neumann, dGamma) \
            - integrate(rhs, dV)

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - exact_sol(x), Gamma1())
            ]


class ConvergenceNeumann2dCircleTest(unittest.TestCase):

    def setUp(self):
        return

    @staticmethod
    def solve(verbose=False):
        def solver(mesh):
            matrix, rhs = pyfvm.discretize_linear(Neumann(), mesh)
            ml = pyamg.smoothed_aggregation_solver(matrix)
            u = ml.solve(rhs, tol=1e-10)
            return u

        return helpers.perform_convergence_tests(
            solver,
            exact_sol,
            helpers.get_circle_mesh,
            range(6),
            verbose=verbose
            )

    def test(self):
        H, error_norm_1, error_norm_inf, order_1, order_inf = self.solve()

        expected_order = 2
        tol = 5.0e-2
        self.assertGreater(order_1[-1], expected_order - tol)
        self.assertGreater(order_inf[-1], expected_order - tol)

        return


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    H, error_norm_1, error_norm_inf, order_1, order_inf = \
        ConvergenceNeumann2dCircleTest.solve(verbose=True)

    helpers.plot_error_data(H, error_norm_1, error_norm_inf)
    plt.show()
