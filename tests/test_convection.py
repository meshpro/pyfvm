import helpers
import meshplex

# import pytest
import meshzoo
import numpy as np
import pyamg
import sympy
from sympy import Matrix, cos, pi, sin

import pyfvm
from pyfvm.form_language import Boundary, dS, dV, integrate, n_dot, n_dot_grad


class Square:
    def exact_sol(self, x):
        return sin(pi * x[0]) * sin(pi * x[1])

    def apply(self, u):
        a0 = 2
        a1 = 1
        a = sympy.Matrix([a0, a1, 0])
        return integrate(lambda x: -n_dot_grad(u(x)) + n_dot(a) * u(x), dS) - integrate(
            lambda x: 2 * pi**2 * sin(pi * x[0]) * sin(pi * x[1])
            + a0 * pi * cos(pi * x[0]) * sin(pi * x[1])
            + a1 * pi * sin(pi * x[0]) * cos(pi * x[1]),
            dV,
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Boundary())]

    def get_mesh(self, k):
        n = 2 ** (k + 1)
        vertices, cells = meshzoo.rectangle(0.0, 1.0, 0.0, 1.0, n + 1, n + 1)
        return meshplex.Mesh(vertices, cells)


class Circle:
    def exact_sol(self, x):
        return cos(pi / 2 * (x[0] ** 2 + x[1] ** 2))

    def apply(self, u):
        a0 = 2
        a1 = 1
        a = np.array([a0, a1, 0])

        def rhs(x):
            z = pi / 2 * (x[0] ** 2 + x[1] ** 2)
            return (
                2 * pi * (sin(z) + z * cos(z))
                - a0 * pi * x[0] * sin(z)
                - a1 * pi * x[1] * sin(z)
            )

        return integrate(lambda x: -n_dot_grad(u(x)) + n_dot(a) * u(x), dS) - integrate(
            rhs, dV
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Boundary())]

    def get_mesh(self, k):
        return helpers.get_disk_mesh(k)


class Cube:
    def exact_sol(self, x):
        return sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])

    def apply(self, u):
        a0 = 2
        a1 = 1
        a2 = 3
        a = Matrix([a0, a1, a2])
        return integrate(lambda x: -n_dot_grad(u(x)) + n_dot(a) * u(x), dS) - integrate(
            lambda x: 3 * pi**2 * sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])
            + a0 * pi * cos(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])
            + a1 * pi * sin(pi * x[0]) * cos(pi * x[1]) * sin(pi * x[2])
            + a2 * pi * sin(pi * x[0]) * sin(pi * x[1]) * cos(pi * x[2]),
            dV,
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Boundary())]

    def get_mesh(self, k):
        n = 2 ** (k + 1)
        vertices, cells = meshzoo.cube(
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, n + 1, n + 1, n + 1
        )
        return meshplex.Mesh(vertices, cells)


class Ball:
    def exact_sol(self, x):
        return cos(pi / 2 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2))

    def apply(self, u):
        a0 = 2
        a1 = 1
        a2 = 3
        a = Matrix([a0, a1, a2])

        def rhs(x):
            z = pi / 2 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
            return (
                +2 * pi * (1.5 * sin(z) + z * cos(z))
                - a0 * pi * x[0] * sin(z)
                - a1 * pi * x[1] * sin(z)
                - a2 * pi * x[2] * sin(z)
            )

        out = integrate(lambda x: -n_dot_grad(u(x)) + n_dot(a) * u(x), dS) - integrate(
            rhs, dV
        )

        return out

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Boundary())]

    def get_mesh(self, k):
        return helpers.get_ball_mesh(k)


def solve(problem, max_k, verbose=False):
    def solver(mesh):
        matrix, rhs = pyfvm.discretize_linear(problem, mesh)
        ml = pyamg.smoothed_aggregation_solver(matrix)
        u = ml.solve(rhs, tol=1e-10)
        return u

    return helpers.perform_convergence_tests(
        solver, problem.exact_sol, problem.get_mesh, range(max_k), verbose=verbose
    )


# TODO turn back on when <https://github.com/sympy/sympy/issues/15071> is resolved
# @pytest.mark.parametrize(
#     "problem, max_k", [(Square(), 6), (Circle(), 4), (Cube(), 4), (Ball(), 3)]
# )
# def test(problem, max_k):
#     H, error_norm_1, error_norm_inf, order_1, order_inf = solve(problem, max_k)
#     expected_order = 2
#     tol = 1.0e-1
#     assert order_1[-1] > expected_order - tol
#     assert order_inf[-1] > expected_order - tol


if __name__ == "__main__":
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(
        Square(), 6, verbose=True
    )
    helpers.show_error_data(H, error_norm_1, error_norm_inf)
