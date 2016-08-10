# -*- coding: utf-8 -*-
#
import numpy
from . import form_language
from .discretize_linear import _discretize_edge_integral
import fvm_problem
import jacobian
import sympy


class EdgeKernel(object):
    def __init__(self, mesh, val):
        self.mesh = mesh
        self.val = val
        self.subdomains = ['everywhere']
        return

    def eval(self, u, edge_ids):
        edge_nodes = self.mesh.edges['nodes'][edge_ids]
        X = self.mesh.node_coords[edge_nodes]
        x0 = X[:, 0, :].T
        x1 = X[:, 1, :].T
        edge_ce_ratio = self.mesh.ce_ratios[edge_ids]
        edge_length = self.mesh.edge_lengths[edge_ids]
        zero = numpy.zeros((2, len(edge_ids)))
        return numpy.array(self.val(
            u[edge_nodes[:, 0]], u[edge_nodes[:, 1]],
            x0, x1, edge_ce_ratio, edge_length
            )) + zero


class VertexKernel(object):
    def __init__(self, mesh, val):
        self.mesh = mesh
        self.val = val
        self.subdomains = ['everywhere']
        return

    def eval(self, u, vertex_ids):
        control_volumes = self.mesh.control_volumes[vertex_ids]
        X = self.mesh.node_coords[vertex_ids].T
        zero = numpy.zeros(len(vertex_ids))
        return self.val(u, control_volumes, X) + zero


class BoundaryKernel(object):
    def __init__(self, mesh, val):
        self.mesh = mesh
        self.val = val
        self.subdomains = ['everywhere']
        return

    def eval(self, u, vertex_ids):
        surface_areas = self.mesh.surface_areas[vertex_ids]
        X = self.mesh.node_coords[vertex_ids].T
        zero = numpy.zeros(len(vertex_ids))
        return self.val(u, x, surface_areas, X) + zero


class DirichletKernel(object):
    def __init__(self, mesh, val, subdomain):
        self.mesh = mesh
        self.val = val
        self.subdomain = subdomain
        return

    def eval(self, u, vertex_ids):
        assert len(u) == len(vertex_ids)
        X = self.mesh.node_coords[vertex_ids].T
        zero = numpy.zeros(len(vertex_ids))
        return self.val(u, X) + zero


def discretize(obj, mesh):
    u = sympy.Function('u')
    res = obj.apply(u)

    # See <http://docs.sympy.org/dev/modules/utilities/lambdify.html>.
    a2a = [{'ImmutableMatrix': numpy.array}, 'numpy']

    edge_kernels = set()
    vertex_kernels = set()
    boundary_kernels = set()

    jacobian_edge_kernels = set()
    jacobian_vertex_kernels = set()
    jacobian_boundary_kernels = set()
    for integral in res.integrals:
        if isinstance(integral.measure, form_language.ControlVolumeSurface):
            # discretization
            x0 = sympy.Symbol('x0')
            x1 = sympy.Symbol('x1')
            el = sympy.Symbol('edge_length')
            er = sympy.Symbol('edge_ce_ratio')
            expr, index_vars = _discretize_edge_integral(
                        integral.integrand, x0, x1, el, er, [u]
                        )
            expr = sympy.simplify(expr)

            # Turn edge around
            uk0 = index_vars[0][0]
            uk1 = index_vars[0][1]
            expr_turned = expr.subs(
                    {uk0: uk1, uk1: uk0, x0: x1, x1: x0},
                    simultaneous=True
                    )

            val = sympy.lambdify(
                (uk0, uk1, x0, x1, er, el), [expr, expr_turned], modules=a2a
                )
            edge_kernels.add(EdgeKernel(mesh, val))

            # Linearization
            expr_lin0 = [sympy.diff(expr, var) for var in [uk0, uk1]]
            expr_lin1 = [sympy.diff(expr_turned, var) for var in [uk0, uk1]]
            val_lin = sympy.lambdify(
                (uk0, uk1, x0, x1, er, el), [expr_lin0, expr_lin1], modules=a2a
                )

            jacobian_edge_kernels.add(EdgeKernel(mesh, val_lin))

        elif isinstance(integral.measure, form_language.ControlVolume):
            x = sympy.DeferredVector('x')
            fx = integral.integrand(x)

            # discretization
            uk0 = sympy.Symbol('uk0')
            try:
                expr = fx.subs(u(x), uk0)
            except AttributeError:  # 'float' object has no
                expr = fx
            control_volume = sympy.Symbol('control_volume')
            expr *= control_volume

            val = sympy.lambdify((uk0, control_volume, x), expr, modules=a2a)

            vertex_kernels.add(VertexKernel(mesh, val))

            # Linearization
            expr_lin = sympy.diff(expr, uk0)
            val_lin = sympy.lambdify(
                (uk0, control_volume, x), expr_lin, modules=a2a
                )
            jacobian_vertex_kernels.add(VertexKernel(mesh, val_lin))

        elif isinstance(integral.measure, form_language.BoundarySurface):
            x = sympy.DeferredVector('x')
            fx = integral.integrand(x)

            # discretization
            uk0 = sympy.Symbol('uk0')
            try:
                expr = fx.subs(u(x), uk0)
            except AttributeError:  # 'float' object has no
                expr = fx
            surface_area = sympy.Symbol('surface_area')
            expr *= surface_area

            val = sympy.lambdify((uk0, surface_area, x), expr, modules=a2a)

            boundary_kernels.add(BoundaryKernel(mesh, val))

            # Linearization
            expr_lin = sympy.diff(expr, uk0)
            val_lin = sympy.lambdify(
                (uk0, surface_area, x), expr_lin, modules=a2a
                )
            jacobian_boundary_kernels.add(BoundaryKernel(mesh, val_lin))

        else:
            raise RuntimeError(
                    'Illegal measure type \'%s\'.' % integral.measure
                    )

    dirichlet_kernels = set()
    jacobian_dirichlet_kernels = set()
    dirichlet = getattr(obj, 'dirichlet', None)
    if callable(dirichlet):
        u = sympy.Function('u')
        x = sympy.DeferredVector('x')
        for f, subdomain in dirichlet(u):
            uk0 = sympy.Symbol('uk0')
            try:
                expr = f(x).subs(u(x), uk0)
            except AttributeError:  # 'float' object has no
                expr = fx

            val = sympy.lambdify((uk0, x), expr, modules=a2a)

            dirichlet_kernels.add(DirichletKernel(mesh, val, subdomain))

            # Linearization
            expr_lin = sympy.diff(expr, uk0)
            val_lin = sympy.lambdify((uk0, x), expr_lin, modules=a2a)
            jacobian_dirichlet_kernels.add(
                    DirichletKernel(mesh, val_lin, subdomain)
                    )

    residual = fvm_problem.FvmProblem(
            mesh,
            edge_kernels, vertex_kernels, boundary_kernels, dirichlet_kernels
            )

    jac = jacobian.Jacobian(
            mesh,
            jacobian_edge_kernels,
            jacobian_vertex_kernels,
            jacobian_boundary_kernels,
            jacobian_dirichlet_kernels
            )

    return residual, jac
