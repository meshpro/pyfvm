# -*- coding: utf-8 -*-
#
import logging
import sympy
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixSymbol


def discretize_edge_integral(integrand, edge, edge_length, edge_covolume):
    discretizer = DiscretizeEdgeIntegral(edge, edge_length, edge_covolume)
    return discretizer.generate(integrand)


debug = False
if debug:
    logging.basicConfig(level=logging.DEBUG)


class DiscretizeEdgeIntegral(object):
    def __init__(self, edge, edge_length, edge_covolume):
        self.arg_translate = {}
        self.x0 = sympy.Symbol('x0')
        self.x1 = sympy.Symbol('x1')
        self.edge = edge
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
        out = out.subs(n, self.edge / self.edge_length)

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
