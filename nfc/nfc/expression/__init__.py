# -*- coding: utf-8 -*-
#
import os
from string import Template
import sympy
from ..helpers import \
        compare_variables, \
        extract_c_expression, \
        sanitize_identifier


class ExpressionCode(object):
    def __init__(self, cls):
        self.class_name = sanitize_identifier(cls.__name__)
        self.cls = cls
        return

    def get_dependencies(self):
        return []

    def get_cxx_class_object(self, dependency_class_objects):
        x = sympy.MatrixSymbol('x', 3, 1)
        if isinstance(self.cls.eval_body, str):
            # The code is specified literally.
            eval_body = self.cls.eval_body
        else:
            result = self.cls.eval(x)
            unused_args, _ = compare_variables(set([x]), [result])
            eval_body = \
                '\n'.join(('(void) %s;' % name) for name in unused_args) \
                + 'return %s;' % extract_c_expression(result)

        # template substitution
        with open('expression.tpl', 'r') as f:
            src = Template(f.read())
            code = src.substitute({
                'name': self.class_name,
                'eval_body': eval_body
                })

        return {
            'code': code,
            'type': 'expression',
            'class_name': self.class_name,
            'constructor_args': []
            }
