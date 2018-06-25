# -*- coding: utf-8 -*-
import meshzoo
import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, dV, Boundary
import numpy
from sympy import exp
import voropy


def test():
    class Bratu(object):
        def apply(self, u):
            return integrate(lambda x: -n_dot_grad(u(x)), dS) - integrate(
                lambda x: 2.0 * exp(u(x)), dV
            )

        def dirichlet(self, u):
            return [(u, Boundary())]

    vertices, cells = meshzoo.rectangle(0.0, 2.0, 0.0, 1.0, 101, 51)
    mesh = voropy.mesh_tri.MeshTri(vertices, cells)

    f, jacobian = pyfvm.discretize(Bratu(), mesh)

    def jacobian_solver(u0, rhs):
        from scipy.sparse import linalg

        jac = jacobian.get_linear_operator(u0)
        return linalg.spsolve(jac, rhs)

    u0 = numpy.zeros(len(vertices))
    u = pyfvm.newton(f.eval, jacobian_solver, u0)
    # import scipy.optimize
    # u = scipy.optimize.newton_krylov(f.eval, u0)

    mesh.write("out.vtk", point_data={"u": u})
    return


if __name__ == "__main__":
    test()
