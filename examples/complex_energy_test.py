import meshplex
import meshzoo
import numpy
import pyamg

import pyfvm


def test():
    class EnergyEdgeKernel:
        def __init__(self):
            self.subdomains = [None]
            return

        def eval(self, mesh, cell_mask):
            edge_ce_ratio = mesh.ce_ratios[..., cell_mask]
            beta = 1.0
            return numpy.array(
                [
                    [edge_ce_ratio, -edge_ce_ratio * numpy.exp(1j * beta)],
                    [-edge_ce_ratio * numpy.exp(-1j * beta), edge_ce_ratio],
                ]
            )

    vertices, cells = meshzoo.rectangle(0.0, 2.0, 0.0, 1.0, 101, 51)
    mesh = meshplex.MeshTri(vertices, cells)

    matrix = pyfvm.get_fvm_matrix(mesh, [EnergyEdgeKernel()], [], [], [])
    rhs = mesh.control_volumes.copy()

    sa = pyamg.smoothed_aggregation_solver(matrix, smooth="energy")
    u = sa.solve(rhs, tol=1e-10)

    # Cannot write complex data ot VTU; split real and imaginary parts first.
    # <http://stackoverflow.com/a/38902227/353337>
    mesh.write("out.vtk", point_data={"u": u.view("(2,)float")})
    return


if __name__ == "__main__":
    test()
