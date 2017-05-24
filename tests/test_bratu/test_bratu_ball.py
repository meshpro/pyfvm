# -*- coding: utf-8 -*-
import helpers

import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, dV, Boundary

import numpy
from sympy import pi, sin, cos, exp


def exact_sol(x):
    return cos(pi/2 * (x[0]**2 + x[1]**2 + x[2]**2))


class Bratu(object):
    def apply(self, u):
        def rhs(x):
            z = pi/2 * (x[0]**2 + x[1]**2 + x[2]**2)
            return 2*pi * (1.5 * sin(z) + z * cos(z)) - 2.0 * exp(cos(z))

        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
            - integrate(lambda x: 2.0 * exp(u(x)), dV) \
            - integrate(rhs, dV)

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - exact_sol(x), Boundary())
            ]


def solve(verbose=False):
    def solver(mesh):
        f, jacobian = pyfvm.discretize(Bratu(), mesh)

        def jacobian_solver(u0, rhs):
            from scipy.sparse import linalg
            jac = jacobian.get_linear_operator(u0)
            return linalg.spsolve(jac, rhs)

        u0 = numpy.zeros(len(mesh.node_coords))
        u = pyfvm.newton(f.eval, jacobian_solver, u0, verbose=True)
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
    tol = 2.0e-1
    assert order_1[-1] > expected_order - tol
    assert order_inf[-1] > expected_order - tol
    return


if __name__ == '__main__':
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(verbose=True)
    helpers.show_error_data(H, error_norm_1, error_norm_inf)
