# -*- coding: utf-8 -*-
import helpers
import numpy
import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, dV, Boundary
import meshzoo
from sympy import pi, sin, exp
import unittest
import voropy


def exact_sol(x):
    return sin(pi*x[0]) * sin(pi*x[1]) * sin(pi*x[2])


class Bratu(object):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
            - integrate(lambda x: 2.0 * exp(u(x)), dV) \
            - integrate(
              lambda x: 3*pi**2 * sin(pi*x[0]) * sin(pi*x[1]) * sin(pi*x[2])
                        - 2.0 * exp(exact_sol(x)),
              dV
              )

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - exact_sol(x), Boundary())
            ]


def get_mesh(k):
    n = 2**(k+1)
    vertices, cells = meshzoo.cube(
            0.0, 1.0,
            0.0, 1.0,
            0.0, 1.0,
            n+1, n+1, n+1
            )
    # return voropy.mesh_tetra.MeshTetra(vertices, cells, mode='algebraic')
    return voropy.mesh_tetra.MeshTetra(vertices, cells, mode='geometric')


class ConvergenceBratu3dCubeTest(unittest.TestCase):

    def setUp(self):
        return

    @staticmethod
    def solve(verbose=False):
        def solver(mesh):
            f, jacobian = pyfvm.discretize(Bratu(), mesh)

            def jacobian_solver(u0, rhs):
                from scipy.sparse import linalg
                jac = jacobian.get_linear_operator(u0)
                return linalg.spsolve(jac, rhs)

            u0 = numpy.zeros(len(mesh.node_coords))
            u = pyfvm.newton(f.eval, jacobian_solver, u0, verbose=False)
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
        ConvergenceBratu3dCubeTest.solve(verbose=True)

    helpers.plot_error_data(H, error_norm_1, error_norm_inf)
    plt.show()