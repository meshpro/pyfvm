# -*- coding: utf-8 -*-
#
import meshzoo
import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, dV, Boundary
from scipy.sparse import linalg
import meshplex


def test():
    class Singular(object):
        def apply(self, u):
            return (
                integrate(lambda x: -1.0e-2 * n_dot_grad(u(x)), dS)
                + integrate(u, dV)
                - integrate(lambda x: 1.0, dV)
            )

        def dirichlet(self, u):
            return [(u, Boundary())]

    vertices, cells = meshzoo.rectangle(0.0, 1.0, 0.0, 1.0, 51, 51)
    mesh = meshplex.MeshTri(vertices, cells)

    matrix, rhs = pyfvm.discretize_linear(Singular(), mesh)

    u = linalg.spsolve(matrix, rhs)

    mesh.write("out.vtk", point_data={"u": u})
    return


if __name__ == "__main__":
    test()
