# -*- coding: utf-8 -*-
#
import os
from string import Template
import sympy
from ..helpers import \
        extract_c_expression, \
        sanitize_identifier_cxx, \
        get_uuid


from pyfvm.form_language import Boundary


class SubdomainCode(object):
    def __init__(self, cls):
        self.cls = cls
        self.class_name_cxx = sanitize_identifier_cxx(cls.__name__)
        self.class_name_python = cls.__name__
        return

    def get_dependencies(self):
        return set()

    def get_cxx_class_object(self, dep_class_objects):
        if self.cls == Boundary:
            # 'Boundary' is already defined
            return {'code': '', 'class_name_cxx': 'boundary'}

        obj = self.cls()

        x = sympy.MatrixSymbol('x', 3, 1)
        result = obj.is_inside(x)

        expr_arguments = set([x])
        try:
            used_vars = result.free_variables
        except AttributeError:
            used_vars = set()
        unused_arguments = expr_arguments - used_vars

        # No undefined variables allowed
        assert(len(used_vars - expr_arguments) == 0)

        try:
            ibo = 'true' if self.cls.is_boundary_only else 'false'
        except AttributeError:
            # AttributeError: 'D2' object has no attribute 'is_boundary_only'
            ibo = 'false'

        # template substitution
        filename = os.path.join(os.path.dirname(__file__), 'subdomain.tpl')
        with open(filename, 'r') as f:
            src = Template(f.read())
            code = src.substitute({
                'name': self.class_name_cxx,
                'id': '"%s"' % self.class_name_cxx,
                'is_inside_return': extract_c_expression(result),
                'is_boundary_only': ibo,
                'is_inside_body': '\n'.join(
                    ('(void) %s;' % name) for name in unused_arguments
                    ),
                })

        return {
            'code': code,
            'class_name_cxx': self.class_name_cxx,
            'type': 'subdomain'
            }

    def get_python_class_object(self, dep_class_objects):
        if self.cls == Boundary:
            # 'Boundary' is already defined
            return {'code': '', 'class_name_cxx': 'boundary'}

        obj = self.cls()

        x = sympy.DeferredVector('x')
        result = obj.is_inside(x)

        expr_arguments = set([x])
        try:
            used_vars = result.free_variables
        except AttributeError:
            used_vars = set()

        # No undefined variables allowed
        assert(len(used_vars - expr_arguments) == 0)

        try:
            ibo = True if self.cls.is_boundary_only else False
        except AttributeError:
            # AttributeError: 'D2' object has no attribute 'is_boundary_only'
            ibo = False

        # template substitution
        filename = os.path.join(os.path.dirname(__file__), 'python.tpl')
        with open(filename, 'r') as f:
            src = Template(f.read())
            code = src.substitute({
                'name': self.class_name_python,
                'id': '\'%s\'' % self.class_name_python,
                'is_inside_return': result,
                'is_boundary_only': ibo
                })

        return {
            'code': code,
            'class_name_cxx': self.class_name_cxx,
            'type': 'subdomain'
            }
