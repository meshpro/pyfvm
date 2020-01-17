import numpy

import meshplex
import meshzoo
import pyfvm
import pykry


def test():
    mu = 5.0e-2
    V = -1.0
    g = 1.0

    class Energy:
        """Specification of the kinetic energy operator.
        """

        def __init__(self):
            self.magnetic_field = mu * numpy.array([0.0, 0.0, 1.0])
            self.subdomains = [None]
            return

        def eval(self, mesh, cell_mask):
            nec = mesh.idx_hierarchy[..., cell_mask]
            X = mesh.node_coords[nec]

            edge_midpoint = 0.5 * (X[0] + X[1])
            edge = X[1] - X[0]
            edge_ce_ratio = mesh.ce_ratios[..., cell_mask]

            # project the magnetic potential on the edge at the midpoint
            magnetic_potential = 0.5 * numpy.cross(self.magnetic_field, edge_midpoint)

            # The dot product <magnetic_potential, edge>, executed for many
            # points at once; cf. <http://stackoverflow.com/a/26168677/353337>.
            beta = numpy.einsum("...k,...k->...", magnetic_potential, edge)

            return numpy.array(
                [
                    [edge_ce_ratio, -edge_ce_ratio * numpy.exp(-1j * beta)],
                    [-edge_ce_ratio * numpy.exp(1j * beta), edge_ce_ratio],
                ]
            )

    vertices, cells = meshzoo.rectangle(-5.0, 5.0, -5.0, 5.0, 51, 51)
    mesh = meshplex.MeshTri(vertices, cells)

    keo = pyfvm.get_fvm_matrix(mesh, edge_kernels=[Energy()])

    def f(psi):
        cv = mesh.control_volumes
        return keo * psi + cv * psi * (V + g * abs(psi) ** 2)

    def jacobian(psi):
        def _apply_jacobian(phi):
            cv = mesh.control_volumes
            y = keo * phi + cv * alpha * phi + cv * gPsi0Squared * phi.conj()
            return y

        alpha = V + g * 2.0 * (psi.real ** 2 + psi.imag ** 2)
        gPsi0Squared = g * psi ** 2

        num_unknowns = len(mesh.node_coords)
        return pykry.LinearOperator(
            (num_unknowns, num_unknowns),
            complex,
            dot=_apply_jacobian,
            dot_adj=_apply_jacobian,
        )

    def jacobian_solver(psi0, rhs):
        jac = jacobian(psi0)
        out = pykry.gmres(
            A=jac,
            b=rhs,
            inner_product=lambda a, b: numpy.dot(a.T.conj(), b).real,
            maxiter=1000,
            tol=1.0e-10,
        )
        return out.xk

    u0 = numpy.ones(len(vertices), dtype=complex)
    u = pyfvm.newton(f, jacobian_solver, u0)

    mesh.write("out.vtk", point_data={"u": u.view("(2,)float")})
    return


if __name__ == "__main__":
    test()
