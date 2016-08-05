# -*- coding: utf-8 -*-
#
import numpy
from .helpers import \
        split_affine_linear_nonlinear, \
        replace_nosh_functions
from . import form_language
from .form_language import n
import linear_fvm_problem
import logging
import sympy
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixSymbol


class EdgeKernel(object):
    def __init__(self, mesh, coeff, affine):
        self.mesh = mesh
        self.coeff = coeff
        self.affine = affine
        self.subdomains = ['everywhere']
        return

    def eval(self, edge_ids):
        X = self.mesh.node_coords[self.mesh.edges['nodes'][edge_ids]]
        x0 = X[:, 0, :].T
        x1 = X[:, 1, :].T
        zero = numpy.zeros(len(edge_ids))
        edge_ce_ratio = self.mesh.ce_ratios[edge_ids]
        edge_length = self.mesh.edge_lengths[edge_ids]
        val = numpy.array(self.coeff(x0, x1, edge_ce_ratio, edge_length, zero))
        # if hasattr(val[0][0], '__len__'):
        #     assert len(val[0][0]) == 1
        #     val = [
        #         [val[0][0][0], val[0][1][0]],
        #         [val[1][0][0], val[1][1][0]]
        #         ]
        return (
            val,
            numpy.array(self.affine(x0, x1, edge_ce_ratio, edge_length, zero))
            )


class VertexKernel(object):
    def __init__(self, mesh, coeff, affine):
        self.mesh = mesh
        self.coeff = coeff
        self.affine = affine
        self.subdomains = ['everywhere']
        return

    def eval(self, vertex_ids):
        control_volumes = self.mesh.control_volumes[vertex_ids]
        X = self.mesh.node_coords[vertex_ids].T
        zero = numpy.zeros(len(vertex_ids))
        return (
            self.coeff(control_volumes, X, zero),
            self.affine(control_volumes, X, zero)
            )


class BoundaryKernel(object):
    def __init__(self, mesh, coeff, affine):
        self.mesh = mesh
        self.coeff = coeff
        self.affine = affine
        self.subdomains = ['everywhere']
        return

    def eval(self, vertex_ids):
        surface_areas = self.mesh.surface_areas[vertex_ids]
        X = self.mesh.node_coords[vertex_ids].T
        zero = numpy.zeros(len(vertex_ids))
        return (
            self.coeff(surface_areas, X, zero),
            self.affine(surface_areas, X, zero)
            )


class DirichletKernel(object):
    def __init__(self, mesh, val, subdomain):
        self.mesh = mesh
        self.val = val
        self.subdomain = subdomain
        return

    def eval(self, vertex_ids):
        X = self.mesh.node_coords[vertex_ids].T
        zero = numpy.zeros(len(vertex_ids))
        return self.val(X, zero)


def _discretize_edge_integral(integrand, x0, x1, edge_length, edge_ce_ratio):
    discretizer = DiscretizeEdgeIntegral(x0, x1, edge_length, edge_ce_ratio)
    return discretizer.generate(integrand)


debug = False
if debug:
    logging.basicConfig(level=logging.DEBUG)


class DiscretizeEdgeIntegral(object):
    def __init__(self, x0, x1, edge_length, edge_ce_ratio):
        self.arg_translate = {}
        self.x0 = x0
        self.x1 = x1
        self.edge_length = edge_length
        self.edge_ce_ratio = edge_ce_ratio
        return

    def visit(self, node):
        if isinstance(node, int):
            return node
        elif isinstance(node, float):
            return node
        elif isinstance(node, sympy.Basic):
            if node.is_Add:
                return self.visit_ChainOp(node, sympy.Add)
            elif node.is_Mul:
                return self.visit_ChainOp(node, sympy.Mul)
            elif node.is_Number:
                return node
            elif node.is_Symbol:
                return node
            elif node.is_Function:
                return self.visit_Call(node)
            elif isinstance(node, MatrixExpr):
                return node

        raise RuntimeError('Unknown node type \"', type(node), '\".')

    def generate(self, node):
        '''Entrance point to this class.
        '''
        x = sympy.MatrixSymbol('x', 3, 1)
        expr = node(x)
        # Collect all function variables.
        function_vars = []
        for f in expr.atoms(sympy.Function):
            if hasattr(f, 'nosh'):
                function_vars.append(f.func)

        out = self.edge_ce_ratio * self.edge_length * self.visit(expr)

        vector_vars = []
        for f in function_vars:
            # Replace f(x0) by f[k0], f(x1) by f[k1].
            k0 = sympy.Symbol('k0')
            k1 = sympy.Symbol('k1')
            f_vec = sympy.IndexedBase('%s' % f)
            out = out.subs(f(self.x0), f_vec[k0])
            out = out.subs(f(self.x1), f_vec[k1])
            # Replace f(x) by 0.5*(f[k0] + f[k1]) (the edge midpoint)
            out = out.subs(f(x), 0.5 * (f_vec[k0] + f_vec[k1]))

            vector_vars.append(f_vec)

        # Replace x by 0.5*(x0 + x1) (the edge midpoint)
        out = out.subs(x, 0.5 * (self.x0 + self.x1))

        # Replace n by the normalized edge
        out = out.subs(n, (self.x1 - self.x0) / self.edge_length)

        return out, vector_vars

    def generic_visit(self, node):
        raise RuntimeError(
            'Should never be called. __name__:', type(node).__name__
            )

    def visit_Load(self, node):
        logging.debug('> Load >')

    def visit_Call(self, node):
        '''Handles calls for operators A(u) and pointwise functions sin(u).
        '''
        try:
            ident = node.func.__name__
        except AttributeError:
            ident = repr(node)
        logging.debug('> Call %s' % ident)
        # Handle special functions
        if ident == 'dot':
            assert(len(node.args) == 2)
            assert(isinstance(node.args[0], MatrixExpr))
            assert(isinstance(node.args[1], MatrixExpr))
            arg0 = self.visit(node.args[0])
            arg1 = self.visit(node.args[1])
            out = node.func(arg0, arg1)
        elif ident == 'n_dot_grad':
            assert(len(node.args) == 1)
            fx = node.args[0]
            f = fx.func
            assert(len(fx.args) == 1)
            assert(isinstance(fx.args[0], MatrixSymbol))
            out = (f(self.x1) - f(self.x0)) / self.edge_length
        else:
            # Default function handling: Assume one argument, e.g., A(x).
            assert(len(node.args) == 1)
            arg = self.visit(node.args[0])
            out = node.func(arg)
        logging.debug('  Call >')
        return out

    def visit_ChainOp(self, node, operator):
        '''Handles binary operations (e.g., +, -, *,...).
        '''
        logging.debug('> BinOp %s' % operator)
        # collect the pointwise code for left and right
        args = []
        for n in node.args:
            ret = self.visit(n)
            args.append(ret)
        # plug it together
        ret = operator(args[0], args[1])
        for k in range(2, len(args)):
            ret = operator(ret, args[k])
        return ret


def _discretize_expression(expr, multiplier=1.0):
    expr, fks = replace_nosh_functions(expr)
    return multiplier * expr, fks


def discretize_linear(obj, mesh):
    u = sympy.Function('u')
    u.nosh = True  # TODO get rid

    res = obj.apply(u)

    # See <http://docs.sympy.org/dev/modules/utilities/lambdify.html>.
    array2array = [{'ImmutableMatrix': numpy.array}, 'numpy']

    zero = sympy.Symbol('zero')

    edge_kernels = set()
    vertex_kernels = set()
    boundary_kernels = set()
    for integral in res.integrals:
        if isinstance(integral.measure, form_language.ControlVolumeSurface):
            x0 = sympy.Symbol('x0')
            x1 = sympy.Symbol('x1')
            edge_length = sympy.Symbol('edge_length')
            edge_ce_ratio = sympy.Symbol('edge_ce_ratio')
            expr, vector_vars = _discretize_edge_integral(
                    integral.integrand,
                    x0, x1,
                    edge_length,
                    edge_ce_ratio
                    )

            u = sympy.IndexedBase('%s' % u)
            k0 = sympy.Symbol('k0')
            k1 = sympy.Symbol('k1')
            uk0 = sympy.Symbol('uk0')
            uk1 = sympy.Symbol('uk1')
            expr = expr.subs([(u[k0], uk0), (u[k1], uk1)])
            #
            expr = sympy.simplify(expr)
            affine0, linear0, nonlinear = \
                split_affine_linear_nonlinear(expr, [uk0, uk1])
            assert nonlinear == 0

            # Turn edge around, do it again
            expr_turned = expr.subs(
                    {uk0: uk1, uk1: uk0, x0: x1, x1: x0},
                    simultaneous=True
                    )
            affine1, linear1, nonlinear = \
                split_affine_linear_nonlinear(expr_turned, [uk0, uk1])
            assert nonlinear == 0

            # Add "zero" to all entities. This later gets translated into
            # np.zeros with the appropriate length, making sure that scalar
            # terms in the lambda expression correctly return np.arrays.
            coeff = [
                [linear0[0] + zero, linear0[1] + zero],
                [linear1[0] + zero, linear1[1] + zero]
                ]
            affine = [
                affine0 + zero,
                affine1 + zero
                ]

            edge_kernels.add(
                EdgeKernel(
                    mesh,
                    sympy.lambdify(
                        (x0, x1, edge_ce_ratio, edge_length, zero),
                        coeff,
                        modules=array2array
                        ),
                    sympy.lambdify(
                        (x0, x1, edge_ce_ratio, edge_length, zero),
                        affine,
                        modules=array2array
                        )
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
            affine, coeff, nonlinear = split_affine_linear_nonlinear(expr, uk0)
            assert nonlinear == 0

            # Add "zero" to all entities. This later gets translated into
            # np.zeros with the appropriate length, making sure that scalar
            # terms in the lambda expression correctly return np.arrays.
            coeff += zero
            affine += zero

            vertex_kernels.add(
                VertexKernel(
                    mesh,
                    sympy.lambdify(
                        (control_volume, x, zero),
                        coeff,
                        modules=array2array
                        ),
                    sympy.lambdify(
                        (control_volume, x, zero),
                        affine,
                        modules=array2array
                        )
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
            affine, coeff, nonlinear = split_affine_linear_nonlinear(expr, uk0)
            assert nonlinear == 0

            # Add "zero" to all entities. This later gets translated into
            # np.zeros with the appropriate length, making sure that scalar
            # terms in the lambda expression correctly return np.arrays.
            coeff += zero
            affine += zero

            boundary_kernels.add(
                BoundaryKernel(
                    mesh,
                    sympy.lambdify(
                        (surface_area, x, zero),
                        coeff,
                        modules=array2array
                        ),
                    sympy.lambdify(
                        (surface_area, x, zero),
                        affine,
                        modules=array2array
                        )
                    )
                )
        else:
            raise RuntimeError(
                    'Illegal measure type \'%s\'.' % integral.measure
                    )

    dirichlet_kernels = set()
    dirichlet = getattr(obj, 'dirichlet', None)
    if callable(dirichlet):
        u = sympy.Function('u')
        x = sympy.DeferredVector('x')
        for f, subdomain in dirichlet(u):
            expr, vector_vars = _discretize_expression(f(x))
            u = sympy.IndexedBase('%s' % u)
            k0 = sympy.Symbol('k')
            uk0 = sympy.Symbol('uk0')
            expr = expr.subs([(u[k0], uk0)])
            affine, coeff, nonlinear = split_affine_linear_nonlinear(expr, uk0)
            assert nonlinear == 0
            rhs = - affine / coeff + zero
            dirichlet_kernels.add(
                    DirichletKernel(
                        mesh,
                        sympy.lambdify(
                            (x, zero),
                            rhs,
                            modules=array2array
                            ),
                        subdomain
                        )
                    )

    return linear_fvm_problem.LinearFvmProblem(
            mesh,
            edge_kernels, vertex_kernels, boundary_kernels, dirichlet_kernels
            )
