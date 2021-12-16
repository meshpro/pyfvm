import meshplex
import meshzoo
import numpy as np
from sympy import exp

import pyfvm
from pyfvm.form_language import Boundary, dS, dV, integrate, n_dot_grad


def test():
    class Bratu:
        def apply(self, u):
            return integrate(lambda x: -n_dot_grad(u(x)), dS) - integrate(
                lambda x: 2.0 * exp(u(x)), dV
            )

        def dirichlet(self, u):
            return [(u, Boundary())]

    vertices, cells = meshzoo.rectangle_tri(
        np.linspace(0.0, 2.0, 101), np.linspace(0.0, 1.0, 51)
    )
    mesh = meshplex.Mesh(vertices, cells)

    f, jac_u = pyfvm.discretize(Bratu(), mesh)

    def jacobian_solver(u0, rhs):
        from scipy.sparse import linalg

        jac = jac_u.get_linear_operator(u0)
        return linalg.spsolve(jac, rhs)

    u0 = np.zeros(len(vertices))
    u = pyfvm.newton(lambda u: f.eval(u), jacobian_solver, u0)
    # import scipy.optimize
    # u = scipy.optimize.newton_krylov(f_eval, u0)

    mesh.write("out.vtk", point_data={"u": u})


if __name__ == "__main__":
    test()
