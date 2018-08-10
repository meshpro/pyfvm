# -*- coding: utf-8 -*-
import helpers

import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, dGamma, dV, Subdomain

import meshzoo
import pyamg
import pytest
from sympy import sin, pi, cos, sqrt
import meshplex


# Everything except the north Boundary()
class Gamma1(Subdomain):
    def is_inside(self, x):
        return x[1] < 1.0 - 1.0e-10

    is_boundary_only = True


class Square(object):
    def exact_sol(self, x):
        return sin(pi * x[0]) * sin(pi * x[1])

    def apply(self, u):
        return (
            integrate(lambda x: -n_dot_grad(u(x)), dS)
            - integrate(lambda x: -pi * sin(pi * x[0]), dGamma)
            - integrate(lambda x: 2 * pi ** 2 * sin(pi * x[0]) * sin(pi * x[1]), dV)
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Gamma1())]

    def get_mesh(self, k):
        n = 2 ** (k + 1)
        vertices, cells = meshzoo.rectangle(
            0.0, 1.0, 0.0, 1.0, n + 1, n + 1, zigzag=True
        )
        return meshplex.MeshTri(vertices, cells)


class Gamma2(Subdomain):
    def is_inside(self, x):
        return x[1] < 0.0

    is_boundary_only = True


class Circle(object):
    def exact_sol(self, x):
        return cos(pi / 2 * (x[0] ** 2 + x[1] ** 2))

    def apply(self, u):
        def neumann(x):
            z = x[0] ** 2 + x[1] ** 2
            return -pi * sqrt(z) * sin(pi / 2 * z)

        def rhs(x):
            z = pi / 2 * (x[0] ** 2 + x[1] ** 2)
            return 2 * pi * (sin(z) + z * cos(z))

        return (
            integrate(lambda x: -n_dot_grad(u(x)), dS)
            - integrate(neumann, dGamma)
            - integrate(rhs, dV)
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Gamma2())]

    def get_mesh(self, k):
        return helpers.get_circle_mesh(k)


def solve(problem, max_k, verbose=False):
    def solver(mesh):
        matrix, rhs = pyfvm.discretize_linear(problem, mesh)
        ml = pyamg.smoothed_aggregation_solver(matrix)
        u = ml.solve(rhs, tol=1e-10)
        return u

    return helpers.perform_convergence_tests(
        solver, problem.exact_sol, problem.get_mesh, range(max_k), verbose=verbose
    )


@pytest.mark.parametrize("problem, max_k", [(Square(), 6), (Circle(), 4)])
def test(problem, max_k):
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(problem, max_k)
    expected_order = 2
    tol = 1.0e-2
    assert order_1[-1] > expected_order - tol
    assert order_inf[-1] > expected_order - tol
    return


if __name__ == "__main__":
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(
        Square(), 6, verbose=True
    )
    helpers.show_error_data(H, error_norm_1, error_norm_inf)
