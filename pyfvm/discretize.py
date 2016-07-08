# -*- coding: utf-8 -*-
#
import numpy
import sympy
from .helpers import \
        extract_linear_components, \
        is_affine_linear, \
        replace_nosh_functions
from . import form_language
from .form_language import n, neg_n, Expression
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
        x0 = X[:, 0, :]
        x1 = X[:, 1, :]
        zero = numpy.zeros(len(X))
        edge_covolume = self.mesh.covolumes[edge_ids]
        edge_length = self.mesh.edge_lengths[edge_ids]
        val = numpy.array(self.coeff(x0, x1, edge_covolume, edge_length, zero))
        # if hasattr(val[0][0], '__len__'):
        #     assert len(val[0][0]) == 1
        #     val = [
        #         [val[0][0][0], val[0][1][0]],
        #         [val[1][0][0], val[1][1][0]]
        #         ]
        return (
            val,
            numpy.array(self.affine(x0, x1, edge_covolume, edge_length, zero))
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
        X = self.mesh.node_coords[vertex_ids]
        zero = numpy.zeros(len(vertex_ids))
        return (
            self.coeff(surface_areas, X, zero),
            self.affine(surface_areas, X, zero)
            )


class DirichletKernel(object):
    def __init__(self, mesh, val, subdomains):
        self.mesh = mesh
        self.val = val
        self.subdomains = subdomains
        return

    def eval(self, vertex_ids):
        X = self.mesh.node_coords[vertex_ids].T
        return self.val(X)


def _discretize_edge_integral(integrand, x0, x1, edge_length, edge_covolume):
    discretizer = DiscretizeEdgeIntegral(x0, x1, edge_length, edge_covolume)
    return discretizer.generate(integrand)


debug = False
if debug:
    logging.basicConfig(level=logging.DEBUG)


class DiscretizeEdgeIntegral(object):
    def __init__(self, x0, x1, edge_length, edge_covolume):
        self.arg_translate = {}
        self.x0 = x0
        self.x1 = x1
        self.edge_length = edge_length
        self.edge_covolume = edge_covolume
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

        out = self.edge_covolume * self.visit(expr)

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
        n = sympy.MatrixSymbol('n', 3, 1)
        out = out.subs(n, (self.x1 - self.x0) / self.edge_length)

        return out, vector_vars

    def generic_visit(self, node):
        raise RuntimeError(
            'Should never be called. __name__:', type(node).__name__
            )
        self.visit(node)

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


def _swap(expr, a, b):
    expr = expr.subs({a: b, b: a}, simultaneous=True)
    return expr


def _extract_linear_components(expr, dvars):
    # TODO replace by helpers.extract_linear_components?
    # Those are the variables in the expression, inserted by the edge
    # discretizer.
    assert is_affine_linear(expr, dvars)

    # Get the coefficients of u0, u1.
    coeff00 = sympy.diff(expr, dvars[0])
    coeff01 = sympy.diff(expr, dvars[1])

    x0 = sympy.Symbol('x0')
    x1 = sympy.Symbol('x1')
    # Now construct the coefficients for the other way around.
    coeff10 = coeff01
    coeff10 = _swap(coeff10, dvars[0], dvars[1])
    coeff10 = _swap(coeff10, x0, x1)
    coeff11 = coeff00
    coeff11 = _swap(coeff11, dvars[0], dvars[1])
    coeff11 = _swap(coeff11, x0, x1)

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

    # See <http://docs.sympy.org/dev/modules/utilities/lambdify.html>.
    array2array = [{'ImmutableMatrix': numpy.array}, 'numpy']

    edge_kernels = set()
    vertex_kernels = set()
    boundary_kernels = set()
    dirichlet_kernels = set()
    for integral in res.integrals:
        if isinstance(integral.measure, form_language.ControlVolumeSurface):
            x0 = sympy.Symbol('x0')
            x1 = sympy.Symbol('x1')
            edge_length = sympy.Symbol('edge_length')
            edge_covolume = sympy.Symbol('edge_covolume')
            expr, vector_vars = _discretize_edge_integral(
                    integral.integrand,
                    x0, x1,
                    edge_length,
                    edge_covolume
                    )
            coeff, affine, arguments, used_vars = _collect_variables(expr, u)

            # Add "zero" to all entities. This later gets translated into
            # np.zeros with the appropriate length, making sure that scalar
            # terms in the lambda expression correctly return np.arrays.
            zero = sympy.Symbol('zero')
            coeff[0][0] += zero
            coeff[0][1] += zero
            coeff[1][0] += zero
            coeff[1][1] += zero
            affine[0] += zero
            affine[1] += zero

            edge_kernels.add(
                EdgeKernel(
                    mesh,
                    sympy.lambdify(
                        (x0, x1, edge_covolume, edge_length, zero),
                        coeff,
                        modules=array2array
                        ),
                    sympy.lambdify(
                        (x0, x1, edge_covolume, edge_length, zero),
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
            coeff, affine = extract_linear_components(expr, uk0)

            # Add "zero" to all entities. This later gets translated into
            # np.zeros with the appropriate length, making sure that scalar
            # terms in the lambda expression correctly return np.arrays.
            zero = sympy.Symbol('zero')
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
            coeff, affine = extract_linear_components(expr, uk0)

            # Add "zero" to all entities. This later gets translated into
            # np.zeros with the appropriate length, making sure that scalar
            # terms in the lambda expression correctly return np.arrays.
            zero = sympy.Symbol('zero')
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
