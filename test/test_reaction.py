# -*- coding: utf-8 -*-
import helpers

import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, dV, Boundary

import pyamg
import pytest
import meshzoo
from sympy import pi, sin, cos
import voropy


class Square(object):
    def exact_sol(self, x):
        return sin(pi * x[0]) * sin(pi * x[1])

    def apply(self, u):
        return (
            integrate(lambda x: -n_dot_grad(u(x)), dS)
            + integrate(lambda x: u(x), dV)
            - integrate(
                lambda x: (2 * pi ** 2 + 1) * sin(pi * x[0]) * sin(pi * x[1]), dV
            )
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Boundary())]

    def get_mesh(self, k):
        n = 2 ** (k + 1)
        vertices, cells = meshzoo.rectangle(
            0.0, 1.0, 0.0, 1.0, n + 1, n + 1, zigzag=True
        )
        return voropy.mesh_tri.MeshTri(vertices, cells)


class Circle(object):
    def exact_sol(self, x):
        return cos(pi / 2 * (x[0] ** 2 + x[1] ** 2))

    def apply(self, u):
        def rhs(x):
            z = pi / 2 * (x[0] ** 2 + x[1] ** 2)
            return 2 * pi * (sin(z) + z * cos(z)) + cos(z)

        return (
            integrate(lambda x: -n_dot_grad(u(x)), dS)
            + integrate(lambda x: u(x), dV)
            - integrate(rhs, dV)
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Boundary())]

    def get_mesh(self, k):
        return helpers.get_circle_mesh(k)


class Cube(object):
    def exact_sol(self, x):
        return sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])

    def apply(self, u):
        def rhs(x):
            return (3 * pi ** 2 + 1) * sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])

        return (
            integrate(lambda x: -n_dot_grad(u(x)), dS)
            + integrate(lambda x: u(x), dV)
            - integrate(rhs, dV)
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Boundary())]

    def get_mesh(self, k):
        n = 2 ** (k + 1)
        vertices, cells = meshzoo.cube(
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, n + 1, n + 1, n + 1
        )
        return voropy.mesh_tetra.MeshTetra(vertices, cells, mode="algebraic")


class Ball(object):
    def exact_sol(self, x):
        return cos(pi / 2 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2))

    def apply(self, u):
        def rhs(x):
            z = pi / 2 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
            return 2 * pi * (1.5 * sin(z) + z * cos(z)) + cos(z)

        return (
            integrate(lambda x: -n_dot_grad(u(x)), dS)
            + integrate(lambda x: u(x), dV)
            - integrate(rhs, dV)
        )

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


@pytest.mark.parametrize(
    "problem, max_k", [(Square(), 6), (Circle(), 4), (Cube(), 4), (Ball(), 3)]
)
def test(problem, max_k):
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(problem, max_k)
    expected_order = 2
    tol = 1.0e-1
    assert order_1[-1] > expected_order - tol
    assert order_inf[-1] > expected_order - tol
    return


if __name__ == "__main__":
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(
        Square(), 6, verbose=True
    )
    helpers.show_error_data(H, error_norm_1, error_norm_inf)
