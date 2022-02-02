import helpers
import meshplex
import meshzoo
import numpy as np
import pytest
from sympy import cos, exp, pi, sin

import pyfvm
from pyfvm.form_language import Boundary, dS, dV, integrate, n_dot_grad


class Square:
    def apply(self, u):
        return (
            integrate(lambda x: -n_dot_grad(u(x)), dS)
            - integrate(lambda x: 2.0 * exp(u(x)), dV)
            - integrate(lambda x: 2 * pi**2 * sin(pi * x[0]) * sin(pi * x[1]), dV)
            + integrate(lambda x: 2.0 * exp(sin(pi * x[0]) * sin(pi * x[1])), dV)
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Boundary())]

    def get_mesh(self, k):
        n = 2 ** (k + 1)
        vertices, cells = meshzoo.rectangle_tri(
            np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, n + 1)
        )
        return meshplex.Mesh(vertices, cells)

    def exact_sol(self, x):
        return sin(pi * x[0]) * sin(pi * x[1])


class Disk:
    def exact_sol(self, x):
        return cos(pi / 2 * (x[0] ** 2 + x[1] ** 2))

    def apply(self, u):
        def rhs(x):
            z = pi / 2 * (x[0] ** 2 + x[1] ** 2)
            return 2 * pi * (sin(z) + z * cos(z)) - 2.0 * exp(cos(z))

        return (
            integrate(lambda x: -n_dot_grad(u(x)), dS)
            - integrate(lambda x: 2.0 * exp(u(x)), dV)
            - integrate(rhs, dV)
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Boundary())]

    def get_mesh(self, k):
        return helpers.get_disk_mesh(k)


class Cube:
    def exact_sol(self, x):
        return sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])

    def apply(self, u):
        return (
            integrate(lambda x: -n_dot_grad(u(x)), dS)
            - integrate(lambda x: 2.0 * exp(u(x)), dV)
            - integrate(
                lambda x: (
                    3 * pi**2 * sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])
                    - 2.0 * exp(self.exact_sol(x))
                ),
                dV,
            )
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Boundary())]

    def get_mesh(self, k):
        n = 2 ** (k + 1)
        vertices, cells = meshzoo.cube_tetra(
            np.linspace(0.0, 1.0, n + 1),
            np.linspace(0.0, 1.0, n + 1),
            np.linspace(0.0, 1.0, n + 1),
        )
        return meshplex.Mesh(vertices, cells)


class Ball:
    def exact_sol(self, x):
        return cos(pi / 2 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2))

    def apply(self, u):
        def rhs(x):
            z = pi / 2 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
            return 2 * pi * (1.5 * sin(z) + z * cos(z)) - 2.0 * exp(cos(z))

        return (
            integrate(lambda x: -n_dot_grad(u(x)), dS)
            - integrate(lambda x: 2.0 * exp(u(x)), dV)
            - integrate(rhs, dV)
        )

    def dirichlet(self, u):
        return [(lambda x: u(x) - self.exact_sol(x), Boundary())]

    def get_mesh(self, k):
        return helpers.get_ball_mesh(k)


def solve(problem, max_k, verbose=False):
    def solver(mesh):
        f, jacobian = pyfvm.discretize(problem, mesh)

        def jacobian_solver(u0, rhs):
            from scipy.sparse import linalg

            jac = jacobian.get_linear_operator(u0)
            return linalg.spsolve(jac, rhs)

        u0 = np.zeros(len(mesh.points))
        u = pyfvm.newton(f.eval, jacobian_solver, u0, verbose=False)
        return u

    return helpers.perform_convergence_tests(
        solver, problem.exact_sol, problem.get_mesh, range(max_k), verbose=verbose
    )


@pytest.mark.parametrize(
    "problem, max_k",
    [
        (Square(), 6),
        (Disk(), 4),
        (Cube(), 4),
        # Disable Ball() to avoid broken gmsh on gh-actions TODO enable
        # (Ball(), 3)
    ],
)
def test(problem, max_k):
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(problem, max_k)
    expected_order = 2
    tol = 5.0e-2
    assert order_1[-1] > expected_order - tol
    assert order_inf[-1] > expected_order - tol


if __name__ == "__main__":
    # problem = Square()
    problem = Disk()
    # problem = Cube()
    # problem = Ball()
    max_k = 6
    H, error_norm_1, error_norm_inf, order_1, order_inf = solve(
        problem, max_k, verbose=True
    )
    helpers.show_error_data(H, error_norm_1, error_norm_inf)
