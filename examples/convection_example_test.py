import numpy
from scipy.sparse import linalg

import meshplex
import meshzoo
import pyfvm
from pyfvm.form_language import Boundary, dS, dV, integrate, n_dot, n_dot_grad


def test():
    class DC:
        def apply(self, u):
            a = numpy.array([2, 1, 0])
            return integrate(
                lambda x: -n_dot_grad(u(x)) + n_dot(a) * u(x), dS
            ) - integrate(lambda x: 1.0, dV)

        def dirichlet(self, u):
            return [(u, Boundary())]

    vertices, cells = meshzoo.rectangle(0.0, 1.0, 0.0, 1.0, 51, 51)
    # TODO get this to work without appending 0
    vertices = numpy.column_stack([vertices, numpy.zeros(len(vertices))])
    mesh = meshplex.MeshTri(vertices, cells)

    matrix, rhs = pyfvm.discretize_linear(DC(), mesh)

    u = linalg.spsolve(matrix, rhs)

    mesh.write("out.vtk", point_data={"u": u})


if __name__ == "__main__":
    test()
