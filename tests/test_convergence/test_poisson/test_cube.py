# -*- coding: utf-8 -*-
import helpers
import pyamg
import pyfvm
from pyfvm.form_language import *
import meshzoo
from sympy import pi, sin
import unittest


def exact_sol(x):
    return sin(pi*x[0]) * sin(pi*x[1]) * sin(pi*x[2])


class Poisson(FvmProblem):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) - \
            integrate(
              lambda x: 3*pi**2 * sin(pi*x[0]) * sin(pi*x[1]) * sin(pi*x[2]),
              dV
              )

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - exact_sol(x), 'boundary')
            ]


def get_mesh(k):
    n = 2**(k+1)
    vertices, cells = meshzoo.cube.create_mesh(
            0.0, 1.0,
            0.0, 1.0,
            0.0, 1.0,
            n+1, n+1, n+1
            )
    # return pyfvm.meshTetra.meshTetra(vertices, cells, mode='algebraic')
    return pyfvm.meshTetra.meshTetra(vertices, cells, mode='geometric')


class ConvergencePoisson3dCubeTest(unittest.TestCase):

    def setUp(self):
        return

    @staticmethod
    def solve(verbose=False):
        def solver(mesh):
            linear_system = pyfvm.discretize_linear(Poisson(), mesh)
            ml = pyamg.ruge_stuben_solver(linear_system.matrix)
            u = ml.solve(linear_system.rhs, tol=1e-10)
            return u

        return helpers.perform_convergence_tests(
            solver,
            exact_sol,
            get_mesh,
            range(4),
            verbose=verbose
            )

    def test(self):
        H, error_norm_1, error_norm_inf, order_1, order_inf = self.solve()

        expected_order = 2
        tol = 1.0e-1
        self.assertGreater(order_1[-1], expected_order - tol)
        self.assertGreater(order_inf[-1], expected_order - tol)

        return


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    H, error_norm_1, error_norm_inf, order_1, order_inf = \
        ConvergencePoisson3dCubeTest.solve(verbose=True)

    helpers.plot_error_data(H, error_norm_1, error_norm_inf)
    plt.show()
