# -*- coding: utf-8 -*-
import helpers

import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, dV, Boundary

import numpy
import pyamg
from sympy import pi, sin, cos
import unittest
import voropy


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


# def get_mesh(k):
#     import dolfin
#     import mshr
#     h = 0.5**(k+2)
#     c = mshr.Sphere(dolfin.Point(0., 0., 0.), 1.0, int(2*pi / h))
#     m = mshr.generate_mesh(c, 2.0 / h)
#     return voropy.mesh_tetra.MeshTetra(
#             m.coordinates(),
#             m.cells(),
#             mode='geometric'
#             )

def get_mesh(k):
    import pygmsh
    h = 0.5**(k+1)
    geom = pygmsh.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, h)
    points, cells, _, _, _ = pygmsh.generate_mesh(geom, verbose=False)
    cells = cells['tetra']
    # toss away unused points
    uvertices, uidx = numpy.unique(cells, return_inverse=True)
    cells = uidx.reshape(cells.shape)
    points = points[uvertices]
    return voropy.mesh_tetra.MeshTetra(points, cells, mode='geometric')


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
            get_mesh,
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
