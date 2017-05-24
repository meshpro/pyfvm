# -*- coding: utf-8 -*-
import helpers

import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, dV, Boundary

import pyamg
from sympy import pi, sin, cos


def exact_sol(x):
    return cos(pi/2 * (x[0]**2 + x[1]**2 + x[2]**2))


class Reaction(object):
    def apply(self, u):
        def rhs(x):
            z = pi/2 * (x[0]**2 + x[1]**2 + x[2]**2)
            return 2*pi * (1.5 * sin(z) + z * cos(z)) + cos(z)

        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
            + integrate(lambda x: u(x), dV) \
            - integrate(rhs, dV)

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - exact_sol(x), Boundary())
            ]


def solve(verbose=False):
    def solver(mesh):
        matrix, rhs = pyfvm.discretize_linear(Reaction(), mesh)
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


def test():
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve()
    expected_order = 2
    tol = 0.2
    assert order_1[-1] > expected_order - tol
    assert order_inf[-1] > expected_order - tol
    return


if __name__ == '__main__':
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(verbose=True)
    helpers.show_error_data(H, error_norm_1, error_norm_inf)
