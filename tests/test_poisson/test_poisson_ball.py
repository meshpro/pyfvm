# -*- coding: utf-8 -*-
import helpers

import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, dV, Boundary

import pyamg
from sympy import pi, sin, cos
import unittest


def exact_sol(x):
    return cos(pi/2 * (x[0]**2 + x[1]**2 + x[2]**2))


class Poisson(object):
    def apply(self, u):
        def rhs(x):
            z = pi/2 * (x[0]**2 + x[1]**2 + x[2]**2)
            return 2*pi * (1.5 * sin(z) + z * cos(z))

        return integrate(lambda x: -n_dot_grad(u(x)), dS) - \
            integrate(rhs, dV)

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - exact_sol(x), Boundary())
            ]


class ConvergencePoisson3dBallTest(unittest.TestCase):

    def setUp(self):
        return

    @staticmethod
    def solve(verbose=False):
        def solver(mesh):
            matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)
            ml = pyamg.smoothed_aggregation_solver(matrix)
            u = ml.solve(rhs, tol=1e-10)
            return u

        return helpers.perform_convergence_tests(
            solver,
            exact_sol,
            helpers.get_ball_mesh,
            range(3),
            verbose=verbose
            )

    def test(self):
        H, error_norm_1, error_norm_inf, order_1, order_inf = self.solve()

        expected_order = 2
        tol = 2.0e-1
        self.assertGreater(order_1[-1], expected_order - tol)
        self.assertGreater(order_inf[-1], expected_order - tol)

        return


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    H, error_norm_1, error_norm_inf, order_1, order_inf = \
        ConvergencePoisson3dBallTest.solve(verbose=True)

    helpers.plot_error_data(H, error_norm_1, error_norm_inf)
    plt.show()
