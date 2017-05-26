# -*- coding: utf-8 -*-
import helpers

import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, \
        dS, dGamma, dV, Subdomain

import meshzoo
import pyamg
from sympy import sin, pi
import voropy


def exact_sol(x):
    return sin(pi*x[0]) * sin(pi*x[1])


# Everything except the north Boundary()
class Gamma1(Subdomain):
    def is_inside(self, x):
        return x[1] < 1.0 - 1.0e-10
    is_boundary_only = True


class Neumann(object):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
            - integrate(lambda x: -pi * sin(pi*x[0]), dGamma) \
            - integrate(lambda x: 2*pi**2 * sin(pi*x[0]) * sin(pi*x[1]), dV)

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - exact_sol(x), Gamma1())
            ]


def get_mesh(k):
    n = 2**(k+1)
    vertices, cells = meshzoo.rectangle(
            0.0, 1.0,
            0.0, 1.0,
            n+1, n+1,
            zigzag=True
            )
    return voropy.mesh_tri.MeshTri(vertices, cells)


def solve(verbose=False):
    def solver(mesh):
        matrix, rhs = pyfvm.discretize_linear(Neumann(), mesh)
        ml = pyamg.smoothed_aggregation_solver(matrix)
        u = ml.solve(rhs, tol=1e-10)
        return u

    return helpers.perform_convergence_tests(
        solver,
        exact_sol,
        get_mesh,
        range(6),
        verbose=verbose
        )


def test():
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve()
    expected_order = 2
    tol = 1.0e-2
    assert order_1[-1] > expected_order - tol
    assert order_inf[-1] > expected_order - tol
    return


if __name__ == '__main__':
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(verbose=True)
    helpers.show_error_data(H, error_norm_1, error_norm_inf)
