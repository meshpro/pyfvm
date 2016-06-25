# -*- coding: utf-8 -*-
#
import sympy
from .compiler import form_language
from .compiler.integral_edge.discretize_edge_integral import \
    discretize_edge_integral
from .compiler.helpers import \
        extract_linear_components, \
        is_affine_linear, \
        replace_nosh_functions
from .compiler.form_language import n, neg_n, Expression
import linear_fvm_problem


class EdgeKernel(object):
    def __init__(self, mesh, coeff, affine):
        self.mesh = mesh
        self.coeff = coeff
        self.affine = affine
        self.subdomains = ['everywhere']
        return

    def eval(self, k):
        edge_covolume = self.mesh.covolumes[k]
        edge_length = self.mesh.edge_lengths[k]
        return (
            self.coeff(edge_covolume, edge_length),
            self.affine(edge_covolume, edge_length)
            )


class VertexKernel(object):
    def __init__(self, mesh, coeff, affine):
        self.mesh = mesh
        self.coeff = coeff
        self.affine = affine
        self.subdomains = ['everywhere']
        return

    def eval(self, k):
        control_volume = self.mesh.control_volumes[k]
        x = self.mesh.node_coords[k]
        return (
            self.coeff(control_volume, x),
            self.affine(control_volume, x)
            )


class BoundaryKernel(object):
    def __init__(self, mesh, coeff, affine):
        self.mesh = mesh
        self.coeff = coeff
        self.affine = affine
        self.subdomains = ['everywhere']
        return

    def eval(self, k):
        surface_area = self.mesh.surface_areas[k]
        x = self.mesh.node_coords[k]
        return (
            self.coeff(surface_area, x),
            self.affine(surface_area, x)
            )


class DirichletKernel(object):
    def __init__(self, mesh, val, subdomains):
        self.mesh = mesh
        self.val = val
        self.subdomains = subdomains
        return

    def eval(self, k):
        x = self.mesh.node_coords[k]
        return self.val(x)


def _collect_variables(expr, matrix_var):
    # Unfortunately, it's not too easy to differentiate with respect to an
    # IndexedBase u with indices k0, k1 respectively. For this reason, we'll
    # simply replace u[k0] by a variable uk0, and u[k1] likewise.
    u = sympy.IndexedBase('%s' % matrix_var)
    k0 = sympy.Symbol('k0')
    k1 = sympy.Symbol('k1')
    uk0 = sympy.Symbol('uk0')
    uk1 = sympy.Symbol('uk1')
    expr = expr.subs([(u[k0], uk0), (u[k1], uk1)])
    edge_coeff, edge_affine = \
        _extract_linear_components(expr, [uk0, uk1])

    arguments = set([sympy.Symbol('edge')])

    # gather up all used variables
    used_vars = set()
    for a in [edge_coeff[0][0], edge_coeff[0][1],
              edge_coeff[1][0], edge_coeff[1][1],
              edge_affine[0], edge_affine[1]
              ]:
        used_vars.update(a.free_symbols)

    return edge_coeff, edge_affine, arguments, used_vars


def _extract_linear_components(expr, dvars):
    # TODO replace by helpers.extract_linear_components?
    # Those are the variables in the expression, inserted by the edge
    # discretizer.
    if not is_affine_linear(expr, dvars):
        raise RuntimeError((
            'The given expression\n'
            '    f(x) = %s\n'
            'does not seem to be affine linear in u.')
            % expr(sympy.Symbol('x'))
            )

    # Get the coefficients of u0, u1.
    coeff00 = sympy.diff(expr, dvars[0])
    coeff01 = sympy.diff(expr, dvars[1])

    # Now construct the coefficients for the other way around.
    coeff10 = coeff01.subs([
        (dvars[0], dvars[1]),
        (dvars[1], dvars[0]),
        (n, neg_n)
        ])
    coeff11 = coeff00.subs([
        (dvars[0], dvars[1]),
        (dvars[1], dvars[0]),
        (n, neg_n)
        ])

    affine = expr.subs([(dvars[0], 0), (dvars[1], 0)])

    return (
        [[coeff00, coeff01], [coeff10, coeff11]],
        [affine, affine]
        )


def _discretize_expression(expr, multiplier):
    expr, fks = replace_nosh_functions(expr)
    return multiplier * expr, fks


def discretize(cls, mesh):
    u = sympy.Function('u')
    u.nosh = True  # TODO get rid

    res = cls.apply(u)

    edge_kernels = set()
    vertex_kernels = set()
    boundary_kernels = set()
    dirichlet_kernels = set()
    for integral in res.integrals:
        if isinstance(integral.measure, form_language.ControlVolumeSurface):
            edge_covolume = sympy.Symbol('edge_covolume')
            edge_length = sympy.Symbol('edge_length')
            expr, vector_vars = discretize_edge_integral(
                    integral.integrand,
                    edge_length,
                    edge_covolume
                    )
            coeff, affine, arguments, used_vars = _collect_variables(expr, u)
            edge_kernels.add(
                EdgeKernel(
                    mesh,
                    sympy.lambdify((edge_covolume, edge_length), coeff),
                    sympy.lambdify((edge_covolume, edge_length), affine)
                    )
                )
        elif isinstance(integral.measure, form_language.ControlVolume):
            # Unfortunately, it's not too easy to differentiate with respect to
            # an IndexedBase u with index k. For this reason, we'll simply
            # replace u[k] by a variable uk0.
            # x = sympy.MatrixSymbol('x', 3, 1)
            x = sympy.DeferredVector('x')
            control_volume = sympy.Symbol('control_volume')
            fx = integral.integrand(x)
            expr, vector_vars = _discretize_expression(fx, control_volume)
            u = sympy.IndexedBase('%s' % u)
            k0 = sympy.Symbol('k')
            uk0 = sympy.Symbol('uk0')
            expr = expr.subs([(u[k0], uk0)])
            coeff, affine = extract_linear_components(expr, uk0)
            vertex_kernels.add(
                VertexKernel(
                    mesh,
                    sympy.lambdify((control_volume, x), coeff),
                    sympy.lambdify((control_volume, x), affine)
                    )
                )
        elif isinstance(integral.measure, form_language.BoundarySurface):
            # Unfortunately, it's not too easy to differentiate with respect to
            # an IndexedBase u with index k. For this reason, we'll simply
            # replace u[k] by a variable uk0.
            # x = sympy.MatrixSymbol('x', 3, 1)
            x = sympy.DeferredVector('x')
            surface_area = sympy.Symbol('surface_area')
            fx = integral.integrand(x)
            expr, vector_vars = _discretize_expression(fx, surface_area)
            u = sympy.IndexedBase('%s' % u)
            k0 = sympy.Symbol('k')
            uk0 = sympy.Symbol('uk0')
            expr = expr.subs([(u[k0], uk0)])
            coeff, affine = extract_linear_components(expr, uk0)
            vertex_kernels.add(
                BoundaryKernel(
                    mesh,
                    sympy.lambdify((surface_area, x), coeff),
                    sympy.lambdify((surface_area, x), affine)
                    )
                )
        else:
            raise RuntimeError(
                    'Illegal measure type \'%s\'.' % integral.measure
                    )

    for dirichlet in cls.dirichlet:
        f, subdomains = dirichlet
        if not isinstance(subdomains, list):
            try:
                subdomains = list(subdomains)
            except TypeError:  # TypeError: 'D1' object is not iterable
                subdomains = [subdomains]
        dirichlet_kernels.add(
                DirichletKernel(
                    mesh,
                    f,
                    subdomains
                    )
                )

    return linear_fvm_problem.LinearFvmProblem(
            mesh,
            edge_kernels, vertex_kernels, boundary_kernels, dirichlet_kernels
            )
