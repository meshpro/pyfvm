# -*- coding: utf-8 -*-
import helpers
import pyfvm
from pyfvm.form_language import *
import meshzoo
from sympy import sin, cos
import numpy
from numpy import pi
import unittest


class Convection(LinearFvmProblem):
    @staticmethod
    def apply(u):
        a0 = 2
        a1 = 1
        a = sympy.Matrix([a0, a1, 0])
        return integrate(lambda x: -n_dot_grad(u(x)) + dot(a.T, n)*u(x), dS) \
            - integrate(
                lambda x:
                    2*pi**2 * sin(pi*x[0]) * sin(pi*x[1]) +
                    a0 * pi * cos(pi*x[0]) * sin(pi*x[1]) +
                    a1 * pi * sin(pi*x[0]) * cos(pi*x[1]),
                dV
                )

    dirichlet = [
            (lambda x: 0.0, ['boundary'])
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
    return pyfvm.meshTri.meshTri(vertices, cells)


class ConvergenceConvection2dSquareTest(unittest.TestCase):

    def setUp(self):
        return

    @staticmethod
    def solve(verbose=False):
        return helpers.perform_convergence_tests(
            Convection,
            exact_sol,
            get_mesh,
            range(6),
            verbose=verbose
            )

    def test(self):
        H, error_norm_1, error_norm_inf, order_1, order_inf = self.solve()

        expected_order = 2
        tol = 1.0e-2
        self.assertGreater(order_1[-1], expected_order - tol)
        self.assertGreater(order_inf[-1], expected_order - tol)

        return


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    H, error_norm_1, error_norm_inf, order_1, order_inf = \
        ConvergenceConvection2dSquareTest.solve(verbose=True)

    helpers.plot_error_data(H, error_norm_1, error_norm_inf)
    plt.show()
