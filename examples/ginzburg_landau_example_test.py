import krylov
import meshplex
import meshzoo
import numpy as np
from scipy.sparse.linalg import LinearOperator

import pyfvm


def test():
    mu = 5.0e-2
    V = -1.0
    g = 1.0

    class Energy:
        """Specification of the kinetic energy operator."""

        def __init__(self):
            self.subdomains = [None]

        def eval(self, mesh, cell_mask):
            nec = mesh.idx[-1][..., cell_mask]
            X = mesh.points[nec]

            edge_midpoint = 0.5 * (X[0] + X[1])
            edge_ce_ratio = mesh.ce_ratios[..., cell_mask]

            # project the magnetic potential on the edge at the midpoint
            # magnetic_field = mu * np.array([0.0, 0.0, 1.0])
            # magnetic_potential = 0.5 * np.cross(magnetic_field, edge_midpoint)
            magnetic_potential = (
                0.5
                * mu
                * np.stack([-edge_midpoint[..., 1], edge_midpoint[..., 0]], axis=-1)
            )

            # The dot product <magnetic_potential, edge>, executed for many
            # points at once; cf. <http://stackoverflow.com/a/26168677/353337>.
            edge = X[1] - X[0]
            beta = np.einsum("...k,...k->...", magnetic_potential, edge)

            return np.array(
                [
                    [edge_ce_ratio, -edge_ce_ratio * np.exp(-1j * beta)],
                    [-edge_ce_ratio * np.exp(1j * beta), edge_ce_ratio],
                ]
            )

    vertices, cells = meshzoo.rectangle_tri(
        np.linspace(-5.0, 5.0, 51), np.linspace(-5.0, 5.0, 51)
    )
    mesh = meshplex.Mesh(vertices, cells)

    keo = pyfvm.get_fvm_matrix(mesh, edge_kernels=[Energy()])

    def f(psi):
        cv = mesh.control_volumes
        return keo * psi + cv * psi * (V + g * abs(psi) ** 2)

    def jacobian(psi):
        def _apply_jacobian(phi):
            cv = mesh.control_volumes
            y = keo * phi + cv * alpha * phi + cv * gPsi0Squared * phi.conj()
            return y

        alpha = V + g * 2.0 * (psi.real**2 + psi.imag**2)
        gPsi0Squared = g * psi**2

        num_unknowns = len(mesh.points)
        return LinearOperator(
            shape=(num_unknowns, num_unknowns),
            matvec=_apply_jacobian,
            rmatvec=_apply_jacobian,
            dtype=complex,
        )

    def jacobian_solver(psi0, rhs):
        sol, _ = krylov.gmres(
            jacobian(psi0),
            rhs,
            inner=lambda a, b: np.dot(a.T.conj(), b).real,
            maxiter=1000,
            tol=1.0e-10,
        )
        assert sol is not None
        return sol

    u0 = np.ones(len(vertices), dtype=complex)
    u = pyfvm.newton(f, jacobian_solver, u0)

    mesh.write("out.vtk", point_data={"u": u.view("(2,)float")})
    return


if __name__ == "__main__":
    test()
