# -*- coding: utf-8 -*-
import helpers
import numpy
import pyfvm
from pyfvm.form_language import *
import mshr
import dolfin
from sympy import pi, sin, cos
import unittest


def exact_sol(x):
    return cos(pi/2 * (x[0]**2 + x[1]**2))


class Reaction(LinearFvmProblem):
    def apply(self, u):
        def rhs(x):
            z = pi/2 * (x[0]**2 + x[1]**2)
            return 2*pi * (sin(z) + z * cos(z)) + cos(z)

        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
            + integrate(lambda x: u(x), dV) \
            - integrate(rhs, dV)

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - exact_sol(x), 'boundary')
            ]


def get_mesh(k):
    h = 0.5**k
    # cell_size = 2 * pi / num_boundary_points
    c = mshr.Circle(dolfin.Point(0., 0., 0.), 1, int(2*pi / h))
    # cell_size = 2 * bounding_box_radius / res
    m = mshr.generate_mesh(c, 2.0 / h)
    coords = m.coordinates()
    coords = numpy.c_[coords, numpy.zeros(len(coords))]
    return pyfvm.meshTri.meshTri(coords, m.cells())


class ConvergenceReaction2dCircleTest(unittest.TestCase):

    def setUp(self):
        return

    @staticmethod
    def solve(verbose=False):
        return helpers.perform_convergence_tests(
            Reaction(),
            exact_sol,
            get_mesh,
            range(7),
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
        ConvergenceReaction2dCircleTest.solve(verbose=True)

    helpers.plot_error_data(H, error_norm_1, error_norm_inf)
    plt.show()
