# -*- coding: utf-8 -*-
import krypy
from krypy.linsys import LinearSystem, Gmres
import meshzoo
import numpy
import pyfvm
import voropy


def test():
    mu = 5.0
    V = -1.0
    g = 1.0

    class Energy(pyfvm.EdgeMatrixKernel):
        """Specification of the kinetic energy operator.
        """

        def __init__(self):
            super(Energy, self).__init__()

            self.magnetic_field = mu * numpy.array([0.0, 0.0, 1.0])
            self.subdomains = [None]
            return

        def eval(self, mesh, cell_mask):
            nec = mesh.idx_hierarchy[..., cell_mask]
            X = mesh.node_coords[nec]

            edge_midpoint = 0.5 * (X[0] + X[1])
            edge = X[1] - X[0]
            edge_ce_ratio = mesh.get_ce_ratios()[..., cell_mask]

            # project the magnetic potential on the edge at the midpoint
            magnetic_potential = (
                0.5 * numpy.cross(self.magnetic_field, edge_midpoint.T).T
            )

            # The dot product <magnetic_potential, edge>, executed for many
            # points at once; cf. <http://stackoverflow.com/a/26168677/353337>.
            beta = numpy.einsum("ijk, ijk->ij", magnetic_potential.T, edge.T)

            return numpy.array(
                [
                    [edge_ce_ratio, -edge_ce_ratio * numpy.exp(1j * beta)],
                    [-edge_ce_ratio * numpy.exp(-1j * beta), edge_ce_ratio],
                ]
            )

    vertices, cells = meshzoo.rectangle(0.0, 1.0, 0.0, 1.0, 31, 31)
    mesh = voropy.mesh_tri.MeshTri(vertices, cells)

    # Equivalently, one could have written
    #
    # from pyfvm.form_language import integrate, dV
    # class GinzburgLandau(object):
    #     def apply(self, psi):
    #         return Energy() \
    #             + integrate(lambda x: psi(x) * (V + g * abs(psi(x))**2), dV)
    # f, _ = pyfvm.discretize(GinzburgLandau(), mesh)
    #
    # The Jacobian still has to be specified manually because of its special
    # structure.

    keo = pyfvm.get_fvm_matrix(mesh, [Energy()], [], [], [])

    def f(psi):
        cv = mesh.get_control_volumes()
        return keo * psi + cv * psi * (V + g * abs(psi) ** 2)

    def jacobian(psi):
        def _apply_jacobian(phi):
            cv = mesh.get_control_volumes()
            s = phi.shape
            y = (
                keo * phi
                + cv.reshape(s) * alpha.reshape(s) * phi
                + cv.reshape(s) * gPsi0Squared.reshape(s) * phi.conj()
            )
            return y

        alpha = V + g * 2.0 * (psi.real ** 2 + psi.imag ** 2)
        gPsi0Squared = g * psi ** 2

        num_unknowns = len(mesh.node_coords)
        return krypy.utils.LinearOperator(
            (num_unknowns, num_unknowns),
            complex,
            dot=_apply_jacobian,
            dot_adj=_apply_jacobian,
        )

    def jacobian_solver(psi0, rhs):
        jac = jacobian(psi0)
        linear_system = LinearSystem(
            A=jac,
            b=rhs,
            self_adjoint=True,
            ip_B=lambda a, b: numpy.dot(a.T.conj(), b).real,
        )
        out = Gmres(linear_system, maxiter=1000, tol=1.0e-10)
        return out.xk[:, 0]

    u0 = numpy.ones(len(vertices), dtype=complex)
    u = pyfvm.newton(f, jacobian_solver, u0)

    mesh.write("out.vtu", point_data={"u": u.view("(2,)float")})
    return


if __name__ == "__main__":
    test()
