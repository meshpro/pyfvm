# -*- coding: utf-8 -*-
import helpers
import pyfvm
from pyfvm.form_language import *
import meshzoo
from sympy import sin
import numpy
from numpy import pi
import unittest


class Poisson(LinearFvmProblem):
    @staticmethod
    def apply(u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
            - integrate(lambda x: 2*pi**2 * sin(pi*x[0]) * sin(pi*x[1]), dV)

    dirichlet = [
            (lambda x: 0.0, ['Boundary'])
            ]


def exact_sol(x):
    return numpy.sin(pi*x[0]) * numpy.sin(pi*x[1])


def get_mesh(k):
    n = 2**(k+1)
    vertices, cells = meshzoo.rectangle.create_mesh(
            0.0, 1.0,
            0.0, 1.0,
            n+1, n+1,
            zigzag=True
            )
    h = 1.0 / n
    return pyfvm.meshTri.meshTri(vertices, cells), h


class ConvergencePoisson2dSquareTest(unittest.TestCase):

    def setUp(self):
        return

    def test(self):
        H, error_norm_1, error_norm_inf, order_1, order_inf = \
            helpers.perform_convergence_tests(
                Poisson,
                exact_sol,
                get_mesh,
                range(6),
                do_print=False
                )

        expected_order = 2
        self.assertAlmostEqual(order_1[-1], expected_order, delta=1.0e-2)
        self.assertAlmostEqual(order_inf[-1], expected_order, delta=1.0e-2)

        return


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    H, error_norm_1, error_norm_inf, order_1, order_inf = \
        helpers.perform_convergence_tests(
            Poisson,
            exact_sol,
            get_mesh,
            range(6),
            do_print=True
            )

    helpers.plot_error_data(H, error_norm_1, error_norm_inf)
    plt.show()
