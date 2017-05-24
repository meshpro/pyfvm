# -*- coding: utf-8 -*-
import helpers

import pyfvm
from pyfvm.form_language import integrate, n_dot, n_dot_grad, dS, dV, Boundary

import pyamg
from sympy import pi, sin, cos, Matrix


def exact_sol(x):
    return cos(pi/2 * (x[0]**2 + x[1]**2))


class Convection(object):
    def apply(self, u):
        a0 = 2
        a1 = 1
        a = Matrix([a0, a1, 0])

        def rhs(x):
            z = pi/2 * (x[0]**2 + x[1]**2)
            return 2*pi * (sin(z) + z * cos(z)) - \
                a0 * pi * x[0] * sin(z) - \
                a1 * pi * x[1] * sin(z)

        return integrate(lambda x: -n_dot_grad(u(x)) + n_dot(a)*u(x), dS) \
            - integrate(rhs, dV)

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - exact_sol(x), Boundary())
            ]


def solve(verbose=False):
    def solver(mesh):
        matrix, rhs = pyfvm.discretize_linear(Convection(), mesh)
        ml = pyamg.smoothed_aggregation_solver(matrix)
        u = ml.solve(rhs, tol=1e-10)
        return u

    return helpers.perform_convergence_tests(
        solver,
        exact_sol,
        helpers.get_circle_mesh,
        range(7),
        verbose=verbose
        )


def test():
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve()
    expected_order = 2
    tol = 1.0e-1
    assert order_1[-1] > expected_order - tol
    assert order_inf[-1] > expected_order - tol
    return


if __name__ == '__main__':
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(verbose=True)
    helpers.show_error_data(H, error_norm_1, error_norm_inf)
