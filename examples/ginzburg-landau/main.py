# -*- coding: utf-8 -*-
import meshzoo
import pyfvm
# from pyfvm.form_language import integrate, dV
import numpy
from krypy.linsys import LinearSystem, Gmres
import krypy

mu = 5.0
V = -1.0
g = 1.0


class Energy(pyfvm.EdgeMatrixKernel):
    def __init__(self):
        super(Energy, self).__init__()

        self.magnetic_field = mu * numpy.array([0.0, 0.0, 1.0])
        self.subdomains = ['everywhere']
        return

    def eval(self, mesh, edge_ids):
        X = mesh.node_coords[mesh.edges['nodes'][edge_ids]]
        x0 = X[:, 0, :].T
        x1 = X[:, 1, :].T
        edge_midpoint = 0.5 * (x0 + x1)
        edge = x1 - x0
        edge_ce_ratio = mesh.ce_ratios[edge_ids]

        # project the magnetic potential on the edge at the midpoint
        magnetic_potential = \
            0.5 * numpy.cross(self.magnetic_field, edge_midpoint.T).T

        # The dot product <magnetic_potential, edge>, executed for many points
        # at once; cf. <http://stackoverflow.com/a/26168677/353337>.
        beta = numpy.einsum('ij, ij->i', magnetic_potential.T, edge.T)

        return numpy.array([
            [edge_ce_ratio, -edge_ce_ratio * numpy.exp(1j * beta)],
            [-edge_ce_ratio * numpy.exp(-1j * beta), edge_ce_ratio]
            ])


# class GinzburgLandau(object):
#     def apply(self, psi):
#         return Energy() \
#             + integrate(lambda x: psi(x) * (V + g * abs(psi(x)))**2, dV)

vertices, cells = meshzoo.rectangle.create_mesh(0.0, 1.0, 0.0, 1.0, 31, 31)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

# f, _ = pyfvm.discretize(GinzburgLandau(), mesh)

keo = pyfvm.get_fvm_matrix(mesh, [Energy()], [], [], [])


def f(psi):
    cv = mesh.control_volumes.reshape(psi.shape)
    return keo * psi + cv * psi * (V + g * abs(psi)**2)


def jacobian(psi):
    '''Implements a LinearOperator object that defines the matrix-vector
    multiplication scheme for the Jacobian operator as in

    .. math::
        A \\phi + B \\phi^*

    with

    .. math::
        A &= K + I (V + g \\cdot 2|\\psi|^2),\\\\
        B &= g \\cdot  diag( \\psi^2 ).
    '''
    def _apply_jacobian(phi):
        cv = mesh.control_volumes.reshape(phi.shape)
        y = keo * phi \
            + cv * alpha.reshape(phi.shape) * phi \
            + cv * gPsi0Squared.reshape(phi.shape) * phi.conj()
        return y

    alpha = V + g * 2.0*(psi.real**2 + psi.imag**2)
    gPsi0Squared = g * psi**2

    num_unknowns = len(mesh.node_coords)
    return krypy.utils.LinearOperator(
            (num_unknowns, num_unknowns),
            complex,
            dot=_apply_jacobian,
            dot_adj=_apply_jacobian
            )


def jacobian_solver(psi0, rhs):
    jac = jacobian(psi0)
    linear_system = LinearSystem(
             A=jac,
             b=rhs,
             self_adjoint=True,
             # !!!
             ip_B=lambda a, b: numpy.dot(a.T.conj(), b).real
             )
    out = Gmres(linear_system, maxiter=1000, tol=1.0e-10)
    return out.xk[:, 0]

u0 = numpy.ones(len(vertices), dtype=complex)
u = pyfvm.newton(f, jacobian_solver, u0)

mesh.write('out.vtu', point_data={'u': u.view('(2,)float')})
