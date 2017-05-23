# -*- coding: utf-8 -*-
import helpers
import pyamg
import pyfvm
from pyfvm.form_language import integrate, n_dot, n_dot_grad, dS, dV, Boundary
import mshr
import dolfin
import numpy
from sympy import pi, sin, cos, Matrix
import voropy


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


def get_mesh(k):
    h = 0.5**k
    # cell_size = 2 * pi / num_Boundary()_points
    c = mshr.Circle(dolfin.Point(0., 0., 0.), 1, int(2*pi / h))
    # cell_size = 2 * bounding_box_radius / res
    m = mshr.generate_mesh(c, 2.0 / h)
    coords = m.coordinates()
    coords = numpy.c_[coords, numpy.zeros(len(coords))]
    return voropy.mesh_tri.MeshTri(coords, m.cells())


def solve(verbose=False):
    def solver(mesh):
        matrix, rhs = pyfvm.discretize_linear(Convection(), mesh)
        ml = pyamg.smoothed_aggregation_solver(matrix)
        u = ml.solve(rhs, tol=1e-10)
        return u

    return helpers.perform_convergence_tests(
        solver,
        exact_sol,
        get_mesh,
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
    from matplotlib import pyplot as plt

    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(verbose=True)

    helpers.plot_error_data(H, error_norm_1, error_norm_inf)
    plt.show()
