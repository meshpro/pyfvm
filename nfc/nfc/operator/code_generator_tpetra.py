# -*- coding: utf-8 -*-
#
import logging
import sympy

import nfl


def get_code_tpetra(expr, arg_translate=None):
    if arg_translate is None:
        arg_translate = {}
    gen = TpetraCodeGenerator()
    return gen.generate(expr, arg_translate)


class Pointwise(str):
    pass


class Vector(str):
    pass


class TpetraCodeGenerator(object):
    def __init__(self):  # , operators, vector_args, scalar_args):
        self._arg_translate = {}
        self._intermediate_count = 0
        self._get_data = set([])
        self._code = ''
        self._required_fvm_matrices = []
        return

    def visit(self, node, out_vector=None):
        if isinstance(node, int):
            return Pointwise(node)
        elif isinstance(node, float):
            return Pointwise(node)
        elif isinstance(node, sympy.Basic):
            if node.is_Add:
                return self.visit_ChainOp(node, '+')
            elif node.is_Mul:
                return self.visit_ChainOp(node, '*')
            elif node.is_Number:
                return Pointwise(node)
            elif node.is_Symbol:
                return self.visit_Name(node)
            elif node.is_Function:
                a = self.visit_Call(node)
                return a

        raise RuntimeError('Unknown node type \"', type(node), '\".')
        return

    def generate(self, node, arg_translate):
        '''Entrance point to this class.
        '''
        self_arg_translate = arg_translate
        self._intermediate_count = 0
        self._get_data = set([])
        self._code = ''
        self._required_fvm_matrices = []
        out = self.visit(node, 'y')
        if isinstance(out, Pointwise):
            self._to_vector(out, 'y')
        return self._code, self._required_fvm_matrices

    def _get_outvector(self):
        '''Sometime, one needs to store intermediate values in vectors. This
        function provides a admissible name for an intermediate vector.
        '''
        out_vector = 'y%d' % self._intermediate_count
        self._intermediate_count += 1
        self._code += '\nTpetra::Vector<double> %s(y->getMap());' % out_vector
        return out_vector

    def _to_vector(self, pointwise_code, out_vector):
        '''This method takes pointwise code and wraps it up into a vector. The
        name of the vector containing the data is returned.
        '''
        assert(isinstance(pointwise_code, Pointwise))
        self._code += '''
auto %sData = %s.getDataNonConst();
for (size_t k = 0; k < %sData.size(); k++) {
  %sData[k] = %s;
}
''' % (out_vector, out_vector, out_vector, out_vector, pointwise_code)
        self._get_data.add(out_vector)
        return

    def _to_pointwise(self, name):
        if name not in self._get_data:
            self._code += '\nconst auto %sData = %s.getData();' % (name, name)
            self._get_data.add(name)
        return '%sData[k]' % name

    def generic_visit(self, node):
        raise RuntimeError(
            'Should never be called. __name__:', type(node).__name__
            )
        self.visit(node)
        return

    def visit_Load(self, node):
        logging.debug('> Load >')
        pass

    def visit_Call(self, node):
        '''Handles calls for operators A(u) and pointwise functions sin(u).
        '''
        name = node.func.__name__
        logging.debug('> Call %s' % name)
        # Check if this is the top (or root) of the recursion. If it is, the
        # output variable will be `y`.
        assert(len(node.args) == 1)  # one argument, e.g., A(x)
        ret = self.visit(node.args[0])
        if isinstance(node, nfl.FvmMatrix):
            # The argument to A(.) must be of vector type.
            if isinstance(ret, Vector):
                arg_name = ret
            elif isinstance(ret, Pointwise):
                arg_name = self._get_outvector()
                self._to_vector(ret, arg_name)
            else:
                raise ValueError('Illegal input type')
            # Get the output vector
            if hasattr(node, 'out_vector'):
                out_vector = node.out_vector
            else:
                out_vector = self._get_outvector()
            # Put it all together
            var_name = name.lower() + '_'
            self._code += '\n%s->apply(%s, %s);\n' \
                % (var_name, arg_name, out_vector)
            self._required_fvm_matrices.append({
                'var_name': var_name,
                'class': node.func,
                })
            logging.debug('  Call >')
            return Vector(out_vector)
        else:
            # Assume that the operator is a C++ intrinsic or otherwise defined.
            # The argument must be of pointwise type.
            if isinstance(ret, Vector):
                a = self._to_pointwise(ret)
            elif isinstance(ret, Pointwise):
                a = ret
            else:
                raise ValueError('Illegal input type')
            logging.debug('  Call >')
            return Pointwise('%s(%s)' % (name, a))

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

    def visit_ChainOp(self, node, code_op):
        '''Handles binary operations (e.g., +, -, *,...).
        '''
        logging.debug('> BinOp %s' % code_op)
        # collect the pointwise code for left and right
        pwcode = []
        for n in node.args:
            ret = self.visit(n)
            if isinstance(ret, Vector):
                pwcode.append(self._to_pointwise(ret))
            elif isinstance(ret, Pointwise):
                pwcode.append(ret)
            else:
                raise ValueError('Illegal input type')
        # plug it together
        co = ' ' + code_op + ' '
        # TODO turn "-1 *" into unary operator
        pointwise_code = co.join(pwcode)
        logging.debug('  BinOp >')
        return Pointwise(pointwise_code)

    def visit_Name(self, node):
        id = node.name
        logging.debug('> Name %s >' % id)
        if id in self._arg_translate:
            return Vector(self._arg_translate[id])
        elif isinstance(node, sympy.Symbol):
            # Treat all other symbols as pointwise variables
            return Pointwise(id)
        else:
            raise ValueError(
                'Name \"%s\" not defined. Should it be an argument?' % id
                )

    def visit_Add(self, node):
        return '+'

    def visit_Sub(self, node):
        return '-'

    def visit_Mult(self, node):
        return '*'

    def visit_Div(self, node):
        return '/'

    def visit_UAdd(self, node):
        return '+'

    def visit_USub(self, node):
        return '-'

    def visit_Num(self, node):
        return Pointwise(str(node.n))
