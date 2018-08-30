# -*- coding: utf-8 -*-
#
from . import form_language
from .discretize_linear import _discretize_edge_integral
from . import fvm_problem
from . import fvm_matrix
from . import jacobian

import numpy
import sympy


class EdgeKernel(object):
    def __init__(self, val):
        self.val = val
        self.subdomains = [None]
        return

    def eval(self, u, mesh, cell_ids):
        node_edge_face_cells = mesh.idx_hierarchy[..., cell_ids]
        X = mesh.node_coords[node_edge_face_cells]
        x0 = X[..., 0]
        x1 = X[..., 1]
        edge_ce_ratio = mesh.ce_ratios[..., cell_ids]
        edge_length = numpy.sqrt(mesh.ei_dot_ei[..., cell_ids])
        zero = numpy.zeros(node_edge_face_cells.shape)
        return (
            numpy.array(
                self.val(
                    u[node_edge_face_cells[0]],
                    u[node_edge_face_cells[1]],
                    x0,
                    x1,
                    edge_ce_ratio,
                    edge_length,
                )
            )
            + zero
        )


class VertexKernel(object):
    def __init__(self, val):
        self.val = val
        self.subdomains = [None]
        return

    def eval(self, u, mesh, vertex_ids):
        control_volumes = mesh.control_volumes[vertex_ids]
        X = mesh.node_coords[vertex_ids].T
        zero = numpy.zeros(len(control_volumes))
        return self.val(u, control_volumes, X) + zero


class FaceKernel(object):
    def __init__(self, val, subdomain):
        self.val = val
        self.subdomain = subdomain
        return

    def eval(self, u, mesh, cell_face_nodes):
        face_areas = mesh.get_face_areas(cell_face_nodes)
        X = mesh.node_coords[cell_face_nodes].T
        zero = numpy.zeros(len(cell_face_nodes))
        return self.val(u, face_areas, X) + zero


class DirichletKernel(object):
    def __init__(self, val, subdomain):
        self.val = val
        self.subdomain = subdomain
        return

    def eval(self, u, mesh, vertex_mask):
        assert len(u) == sum(vertex_mask)
        X = mesh.node_coords[vertex_mask].T
        zero = numpy.zeros(sum(vertex_mask))
        return self.val(u, X) + zero


def discretize(obj, mesh):
    u = sympy.Function("u")

    lmbda = sympy.Function("lambda")
    try:
        res = obj.apply(u, lmbda)
    except TypeError:
        res = obj.apply(u)

    # res = obj.apply(u)

    # See <http://docs.sympy.org/dev/modules/utilities/lambdify.html>.
    a2a = [{"ImmutableMatrix": numpy.array}, "numpy"]

    edge_kernels = set()
    vertex_kernels = set()
    face_kernels = set()
    edge_matrix_kernels = set()
    # vertex_matrix_kernels = set()
    # boundary_matrix_kernels = set()

    jacobian_edge_kernels = set()
    jacobian_vertex_kernels = set()
    jacobian_face_kernels = set()

    for integral in res.integrals:
        if isinstance(integral.measure, form_language.ControlVolumeSurface):
            # discretization
            x0 = sympy.Symbol("x0")
            x1 = sympy.Symbol("x1")
            el = sympy.Symbol("edge_length")
            er = sympy.Symbol("edge_ce_ratio")
            expr, index_vars = _discretize_edge_integral(
                integral.integrand, x0, x1, el, er, [u]
            )
            expr = sympy.simplify(expr)

            # Turn edge around
            uk0 = index_vars[0][0]
            uk1 = index_vars[0][1]
            expr_turned = expr.subs(
                {uk0: uk1, uk1: uk0, x0: x1, x1: x0}, simultaneous=True
            )

            val = sympy.lambdify(
                (uk0, uk1, x0, x1, er, el), [expr, expr_turned], modules=a2a
            )
            edge_kernels.add(EdgeKernel(val))

            # Linearization
            expr_lin0 = [sympy.diff(expr, var) for var in [uk0, uk1]]
            expr_lin1 = [sympy.diff(expr_turned, var) for var in [uk0, uk1]]
            val_lin = sympy.lambdify(
                (uk0, uk1, x0, x1, er, el), [expr_lin0, expr_lin1], modules=a2a
            )

            jacobian_edge_kernels.add(EdgeKernel(val_lin))

        elif isinstance(integral.measure, form_language.ControlVolume):
            x = sympy.DeferredVector("x")

            fx = integral.integrand(x)

            # discretization
            uk0 = sympy.Symbol("uk0")
            try:
                expr = fx.subs(u(x), uk0)
            except AttributeError:  # 'float' object has no
                expr = fx
            control_volume = sympy.Symbol("control_volume")
            expr *= control_volume

            val = sympy.lambdify((uk0, control_volume, x), expr, modules=a2a)

            vertex_kernels.add(VertexKernel(val))

            # Linearization
            expr_lin = sympy.diff(expr, uk0)
            val_lin = sympy.lambdify((uk0, control_volume, x), expr_lin, modules=a2a)
            jacobian_vertex_kernels.add(VertexKernel(val_lin))

        else:
            assert isinstance(integral.measure, form_language.CellSurface)
            x = sympy.DeferredVector("x")
            fx = integral.integrand(x)

            # discretization
            uk0 = sympy.Symbol("uk0")
            try:
                expr = fx.subs(u(x), uk0)
            except AttributeError:  # 'float' object has no
                expr = fx
            face_area = sympy.Symbol("face_area")
            expr *= face_area

            val = sympy.lambdify((uk0, face_area, x), expr, modules=a2a)

            face_kernels.add(FaceKernel(val))

            # Linearization
            expr_lin = sympy.diff(expr, uk0)
            val_lin = sympy.lambdify((uk0, face_area, x), expr_lin, modules=a2a)
            jacobian_face_kernels.add(FaceKernel(val_lin))

    dirichlet_kernels = set()
    jacobian_dirichlet_kernels = set()
    dirichlet = getattr(obj, "dirichlet", None)
    if callable(dirichlet):
        u = sympy.Function("u")
        x = sympy.DeferredVector("x")
        for f, subdomain in dirichlet(u):
            uk0 = sympy.Symbol("uk0")
            try:
                expr = f(x).subs(u(x), uk0)
            except AttributeError:  # 'float' object has no
                expr = fx

            val = sympy.lambdify((uk0, x), expr, modules=a2a)

            dirichlet_kernels.add(DirichletKernel(val, subdomain))

            # Linearization
            expr_lin = sympy.diff(expr, uk0)
            val_lin = sympy.lambdify((uk0, x), expr_lin, modules=a2a)
            jacobian_dirichlet_kernels.add(DirichletKernel(val_lin, subdomain))

    residual = fvm_problem.FvmProblem(
        mesh,
        edge_kernels,
        vertex_kernels,
        face_kernels,
        dirichlet_kernels,
        edge_matrix_kernels,
        [],
        [],
    )

    jac = jacobian.Jacobian(
        mesh,
        jacobian_edge_kernels,
        jacobian_vertex_kernels,
        jacobian_face_kernels,
        jacobian_dirichlet_kernels,
    )

    return residual, jac
