# -*- coding: utf-8 -*-
#
import logging
import sympy
from sympy.matrices.expressions.matexpr import \
        MatrixElement, MatrixExpr, MatrixSymbol

from ..form_language import n, neg_n

debug = False
if debug:
    logging.basicConfig(level=logging.DEBUG)


class CodeGeneratorEigen(object):
    '''
    Takes a sympy expression and converts it to C/C++ code. Some special
    functions, such as dot(), are converted to Eigen functions.
    '''
    def __init__(self):  # , operators, vector_args, scalar_args):
        self.arg_translate = {}
        self._discretization = 0
        self._required_operators = []
        self.x0 = sympy.Symbol('x0')
        self.x1 = sympy.Symbol('x1')
        self.u0 = sympy.Symbol('u0')
        self.u1 = sympy.Symbol('u1')
        self.edge_length = sympy.Symbol('edge_length')
        self.covolume = sympy.Symbol('covolume')
        return

    def visit(self, node):
        if isinstance(node, int):
            return str(node)
        elif isinstance(node, float):
            return str(node)
        elif isinstance(node, sympy.Basic):
            if node.is_Add:
                return self.visit_ChainOp(node, '+')
            elif node.is_Mul:
                return self.visit_ChainOp(node, '*')
            elif node.is_Pow:
                return self.visit_Pow(node)
            elif node.is_Number:
                return str(node)
            elif node.is_Symbol:
                return str(node)
            elif node.is_Function:
                return self.visit_Call(node)
            elif isinstance(node, MatrixExpr):
                if node == n:
                    return 'n'
                elif node == neg_n:
                    return '-n'
                else:
                    return 'Eigen::Vector3d(%s,%s,%s)' % \
                            (node[0], node[1], node[2])

        raise RuntimeError('Unknown node type \"', type(node), '\".')
        return

    def generate(self, node):
        '''Entrance point to this class.
        '''
        out = self.visit(node)
        return out, self._required_operators

    def generic_visit(self, node):
        raise RuntimeError(
            'Should never be called. __name__:', type(node).__name__
            )
        self.visit(node)
        return

    def visit_Load(self, node):
        logging.debug('> Load >')

    def visit_Call(self, node):
        '''Handles calls for operators A(u) and pointwise functions sin(u).
        '''
        try:
            id = node.func.__name__
        except AttributeError:
            id = repr(node)
        logging.debug('> Call %s' % id)
        # Handle special functions
        if id == 'dot':
            assert(len(node.args) == 2)
            assert(isinstance(node.args[0], MatrixExpr))
            assert(isinstance(node.args[1], MatrixExpr))
            arg0 = self.visit(node.args[0])
            arg1 = self.visit(node.args[1])
            out = '%s.dot(%s)' % (arg0, arg1)
        else:
            # Default function handling: Assume one argument, e.g., A(x).
            assert(len(node.args) == 1)
            arg = self.visit(node.args[0])
            out = node.func(arg)
        logging.debug('  Call >')
        return out

    def visit_UnaryOp(self, node):
        '''Handles unary operations (e.g., +, -,...).
        '''
        code_op = self.visit(node.op)
        logging.debug('> UnaryOp %s' % code_op)
        ret = self.visit(node.operand)
        if isinstance(ret, Vector):
            code = self._to_pointwise(ret)
        elif isinstance(ret, Pointwise):
            code = ret
        else:
            raise ValueError('Illegal input type')
        # plug it together
        pointwise_code = '%s%s' % (code_op, code)
        logging.debug('  UnaryOp >')
        return Pointwise(pointwise_code)

    def visit_ChainOp(self, node, symbol):
        '''Handles binary operations (e.g., +, -, *,...).
        '''
        logging.debug('> BinOp %s' % symbol)
        # collect the pointwise code for left and right
        args = []
        for n in node.args:
            ret = self.visit(n)
            args.append('(%s)' % ret)
        # plug it together
        ret = symbol.join(args)
        return ret

    def visit_Pow(self, node):
        '''Handles pow(.,.).
        '''
        logging.debug('> Pow')
        assert(len(node.args) == 2)
        power = int(node.args[1])
        if power == 1:
            return self.visit(node.args[0])
        elif power == 0:
            return '1'
        elif power == -1:
            return '1.0/(%s)' % self.visit(node.args[0])
        elif power > 1:
            return 'pow(%s, %s)' % (self.visit(node.args[0]), power)
        else:  # node.args[1] < -1
            return '1.0 / pow(%s, %s)' % (self.visit(node.args[0]), power)
