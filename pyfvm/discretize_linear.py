# -*- coding: utf-8 -*-
#
import numpy
from . import form_language
import linear_fvm_problem
import sympy
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixSymbol


def split(expr, variables):
    '''Split affine, linear, and nonlinear part of of expr w.r.t. variables.
    '''
    if isinstance(expr, float):
        return expr, 0, 0

    input_is_list = True
    if not isinstance(variables, list):
        input_is_list = False
        variables = [variables]

    # See <https://github.com/sympy/sympy/issues/11475> on why we need expand()
    # here.
    expr = expr.expand()
    affine = expr
    linear = []
    for var in variables:
        linear.append(sympy.diff(expr, var).coeff(var, 0))
        affine = affine.coeff(var, n=0)

    # The rest is nonlinear
    nonlinear = expr - affine
    for var, coeff in zip(variables, linear):
        nonlinear -= var * coeff
    nonlinear = sympy.simplify(nonlinear)

    if not input_is_list:
        assert len(linear) == 1
        linear = linear[0]

    return affine, linear, nonlinear


class EdgeLinearKernel(object):
    def __init__(self, linear, affine):
        self.linear = linear
        self.affine = affine
        self.subdomains = ['everywhere']
        return

    def eval(self, mesh, edge_ids):
        X = mesh.node_coords[mesh.edges['nodes'][edge_ids]]
        x0 = X[:, 0, :].T
        x1 = X[:, 1, :].T
        edge_ce_ratio = mesh.ce_ratios[edge_ids]
        edge_length = mesh.edge_lengths[edge_ids]
        # Add "zero" to all entities. This later gets translated into
        # np.zeros with the appropriate length, making sure that scalar
        # terms in the lambda expression correctly return np.arrays.
        zero = numpy.zeros(len(edge_ids))

        val = self.linear(x0, x1, edge_ce_ratio, edge_length)
        val[0][0] += zero
        val[0][1] += zero
        val[1][0] += zero
        val[1][1] += zero
        val = numpy.array(val)

        rhs = self.affine(x0, x1, edge_ce_ratio, edge_length)
        rhs[0] += zero
        rhs[1] += zero
        rhs = numpy.array(rhs)

        return (val, rhs)


class VertexLinearKernel(object):
    def __init__(self, mesh, linear, affine):
        self.mesh = mesh
        self.linear = linear
        self.affine = affine
        self.subdomains = ['everywhere']
        return

    def eval(self, vertex_ids):
        control_volumes = self.mesh.control_volumes[vertex_ids]
        X = self.mesh.node_coords[vertex_ids].T
        zero = numpy.zeros(len(vertex_ids))
        return (
            self.linear(control_volumes, X) + zero,
            self.affine(control_volumes, X) + zero
            )


class BoundaryLinearKernel(object):
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
            self.coeff(surface_areas, X) + zero,
            self.affine(surface_areas, X) + zero
            )


class DirichletLinearKernel(object):
    def __init__(self, mesh, coeff, rhs, subdomain):
        self.mesh = mesh
        self.coeff = coeff
        self.rhs = rhs
        self.subdomain = subdomain
        return

    def eval(self, vertex_ids):
        X = self.mesh.node_coords[vertex_ids].T
        zero = numpy.zeros(len(vertex_ids))
        return (
            self.coeff(X) + zero,
            self.rhs(X) + zero
            )


def _discretize_edge_integral(
        integrand, x0, x1, edge_length, edge_ce_ratio,
        index_functions
        ):
    discretizer = DiscretizeEdgeIntegral(x0, x1, edge_length, edge_ce_ratio)
    return discretizer.generate(integrand, index_functions)


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

    def generate(self, node, index_functions=[]):
        '''Entrance point to this class.
        '''
        x = sympy.MatrixSymbol('x', 3, 1)
        expr = node(x)

        out = self.edge_ce_ratio * self.edge_length * self.visit(expr)

        index_vars = []
        for f in index_functions:
            # Replace f(x0) by f[k0], f(x1) by f[k1].
            fk0 = sympy.Symbol('%sk0' % f)
            fk1 = sympy.Symbol('%sk1' % f)
            out = out.subs(f(self.x0), fk0)
            out = out.subs(f(self.x1), fk1)
            # Replace f(x) by 0.5*(f[k0] + f[k1]) (the edge midpoint)
            out = out.subs(f(x), 0.5 * (fk0 + fk1))

            index_vars.append([fk0, fk1])

        # Replace x by 0.5*(x0 + x1) (the edge midpoint)
        out = out.subs(x, 0.5 * (self.x0 + self.x1))

        # Replace n by the normalized edge
        out = out.subs(form_language.n, (self.x1 - self.x0) / self.edge_length)

        return out, index_vars

    def generic_visit(self, node):
        raise RuntimeError(
            'Should never be called. __name__:', type(node).__name__
            )

    def visit_Load(self, node):
        return

    def visit_Call(self, node):
        '''Handles calls for operators A(u) and pointwise functions sin(u).
        '''
        try:
            ident = node.func.__name__
        except AttributeError:
            ident = repr(node)
        # Handle special functions
        if ident == 'dot':
            assert len(node.args) == 2
            assert isinstance(node.args[0], MatrixExpr)
            assert isinstance(node.args[1], MatrixExpr)
            arg0 = self.visit(node.args[0])
            arg1 = self.visit(node.args[1])
            out = node.func(arg0, arg1)
        elif ident == 'n_dot_grad':
            assert len(node.args) == 1
            fx = node.args[0]
            f = fx.func
            assert len(fx.args) == 1
            assert isinstance(fx.args[0], MatrixSymbol)
            out = (f(self.x1) - f(self.x0)) / self.edge_length
        else:
            # Default function handling: Assume one argument, e.g., A(x).
            assert len(node.args) == 1
            arg = self.visit(node.args[0])
            out = node.func(arg)
        return out

    def visit_ChainOp(self, node, operator):
        '''Handles binary operations (e.g., +, -, *,...).
        '''
        # collect the pointwise code for left and right
        args = []
        for arg in node.args:
            ret = self.visit(arg)
            args.append(ret)
        # plug it together
        ret = operator(args[0], args[1])
        for k in range(2, len(args)):
            ret = operator(ret, args[k])
        return ret


def discretize_linear(obj, mesh):
    u = sympy.Function('u')
    res = obj.apply(u)

    # See <http://docs.sympy.org/dev/modules/utilities/lambdify.html>.
    a2a = [{'ImmutableMatrix': numpy.array}, 'numpy']

    edge_kernels = set()
    vertex_kernels = set()
    boundary_kernels = set()
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

            uk0 = index_vars[0][0]
            uk1 = index_vars[0][1]

            affine0, linear0, nonlinear = split(expr, [uk0, uk1])
            assert nonlinear == 0

            # Turn edge around
            expr_turned = expr.subs(
                    {uk0: uk1, uk1: uk0, x0: x1, x1: x0},
                    simultaneous=True
                    )
            affine1, linear1, nonlinear = split(expr_turned, [uk0, uk1])
            assert nonlinear == 0

            linear = [[linear0[0], linear0[1]], [linear1[0], linear1[1]]]
            affine = [affine0, affine1]

            l_eval = sympy.lambdify((x0, x1, er, el), linear, modules=a2a)
            a_eval = sympy.lambdify((x0, x1, er, el), affine, modules=a2a)

            edge_kernels.add(EdgeLinearKernel(mesh, l_eval, a_eval))

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

            affine, linear, nonlinear = split(expr, uk0)
            assert nonlinear == 0

            l_eval = sympy.lambdify((control_volume, x), linear, modules=a2a)
            a_eval = sympy.lambdify((control_volume, x), affine, modules=a2a)

            vertex_kernels.add(VertexLinearKernel(mesh, l_eval, a_eval))

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

            affine, linear, nonlinear = split(expr, uk0)
            assert nonlinear == 0

            l_eval = sympy.lambdify((surface_area, x), linear, modules=a2a)
            a_eval = sympy.lambdify((surface_area, x), affine, modules=a2a)

            boundary_kernels.add(BoundaryLinearKernel(mesh, l_eval, a_eval))

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
            uk0 = sympy.Symbol('uk0')
            try:
                expr = f(x).subs(u(x), uk0)
            except AttributeError:  # 'float' object has no
                expr = fx

            affine, coeff, nonlinear = split(expr, uk0)
            assert nonlinear == 0

            coeff_eval = sympy.lambdify((x), coeff, modules=a2a)
            rhs_eval = sympy.lambdify((x), -affine, modules=a2a)

            dirichlet_kernels.add(
                DirichletLinearKernel(mesh, coeff_eval, rhs_eval, subdomain)
                )

    return linear_fvm_problem.LinearFvmProblem(
            mesh,
            edge_kernels, vertex_kernels, boundary_kernels, dirichlet_kernels
            )
