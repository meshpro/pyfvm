# -*- coding: utf-8 -*-
#
import os
from string import Template
import sympy

from ..helpers import \
        compare_variables, \
        extract_c_expression, \
        get_uuid, \
        list_unique, \
        replace_nosh_functions, \
        sanitize_identifier_cxx
from ..subdomain import *


class Dirichlet(object):
    def __init__(self, function, subdomains, is_matrix):
        uuid = get_uuid()
        self.class_name_cxx = 'dirichlet_' + uuid
        self.class_name_python = 'Dirichlet' + uuid
        self.function = function
        self.is_matrix = is_matrix

        # TODO
        self.scalar_params = set()
        self.vector_params = set()

        self.dependencies = [SubdomainCode(sd) for sd in subdomains]
        if subdomains:
            self.subdomains = subdomains
        else:
            self.subdomains = [Boundary]
        return

    def get_dependencies(self):
        return self.dependencies

    def get_cxx_class_object(self, dep_class_objects):
        # collect subdomain init code
        init = '{%s}' % ', '.join(
                '"%s"' % sanitize_identifier_cxx(sd.__name__)
                for sd in self.subdomains
                )

        if self.is_matrix:
            code = self._get_code_for_matrix(init)
        else:
            code = self._get_code_for_operator(init)

        return {
            'code': code,
            }

    def _get_code_for_matrix(self, init):
        x = sympy.MatrixSymbol('x', 3, 1)
        vertex = sympy.Symbol('vertex')
        result = self.function(x)
        unused_args, _ = compare_variables(set([vertex]), [result])

        # template substitution
        filename = os.path.join(
                os.path.dirname(__file__),
                'matrix_core_dirichlet.tpl'
                )
        with open(filename, 'r') as f:
            code = Template(f.read()).substitute({
                'name': self.class_name_cxx,
                'init': 'nosh::matrix_core_dirichlet(%s)' % init,
                'eval_return_value': extract_c_expression(result),
                'eval_body':
                    '\n'.join('(void) %s;' % arg for arg in unused_args)
                })
        return code

    def _get_code_for_operator(self, init):
        x = sympy.MatrixSymbol('x', 3, 1)
        vertex = sympy.Symbol('vertex')
        u = sympy.Function('u')
        result = self.function(u, x)
        result, fks = replace_nosh_functions(result)
        arguments = set([vertex, u])

        # unused_args, _ = compare_variables(arguments, [result])

        try:
            free_symbols = result.free_symbols
        except AttributeError:
            free_symbols = set()

        extra_body, extra_init, extra_declare = \
            _get_cxx_extra(arguments, free_symbols)

        init = ['nosh::operator_core_dirichlet(%s)' % init]
        init.extend(extra_init)
        declare = extra_declare

        # template substitution
        filename = os.path.join(
                os.path.dirname(__file__),
                'operator_core_dirichlet.tpl'
                )
        with open(filename, 'r') as f:
            code = Template(f.read()).substitute({
                'name': self.class_name_cxx,
                'init': ',\n'.join(init),
                'declare': ',\n'.join(declare),
                'eval_return_value': extract_c_expression(result),
                'eval_body': '\n'.join(extra_body)
                })
        return code

    def get_python_class_object(self, dep_class_objects):
        # collect subdomain init code
        init = '[%s]' % ', '.join(
                '\'%s\'' % sd.__name__
                for sd in self.subdomains
                )

        assert(self.is_matrix)

        x = sympy.MatrixSymbol('x', 3, 1)
        result = self.function(x)

        # template substitution
        filename = os.path.join(os.path.dirname(__file__), 'python.tpl')
        with open(filename, 'r') as f:
            code = Template(f.read()).substitute({
                'name': self.class_name_python,
                'init_subdomains': init,
                'eval_return_value': result
                })

        return {
            'code': code,
            }


def _get_cxx_extra(arguments, used_variables):
    vertex = sympy.Symbol('vertex')
    unused_arguments = arguments - used_variables
    undefined_symbols = used_variables - arguments

    init = []
    body = []
    declare = []

    u = sympy.Symbol('u')
    if u in undefined_symbols:
        undefined_symbols.remove(u)

    x = sympy.MatrixSymbol('x', 3, 1)
    if x in undefined_symbols:
        init.append('mesh_(mesh)')
        declare.append('const std::shared_ptr<const nosh::mesh> mesh_;')
        init.append('c_data_(mesh->control_volumes()->getData())')
        declare.append('const Teuchos::ArrayRCP<const double> c_data_;')
        body.append('const auto k = this->mesh_->local_index(vertex);')
        body.append('const auto x = this->mesh_->get_coords(vertex);')
        undefined_symbols.remove(x)
        if vertex in unused_arguments:
            unused_arguments.remove(vertex)

    k = sympy.Symbol('k')
    if k in undefined_symbols:
        init.append('mesh_(mesh)')
        declare.append('const std::shared_ptr<const nosh::mesh> mesh_;')
        body.append('const auto k = this->mesh_->local_index(vertex);')
        undefined_symbols.remove(k)
        if vertex in unused_arguments:
            unused_arguments.remove(vertex)

    if len(undefined_symbols) > 0:
        raise RuntimeError(
                'The following symbols are undefined: %s' % undefined_symbols
                )

    # remove double lines
    body = list_unique(body)
    init = list_unique(init)
    declare = list_unique(declare)

    for name in unused_arguments:
        body.insert(0, '(void) %s;' % name)

    return body, init, declare


def _get_python_extra(arguments, used_variables):
    vertex = sympy.Symbol('vertex')
    unused_arguments = arguments - used_variables
    undefined_symbols = used_variables - arguments

    body = []

    u = sympy.Symbol('u')
    if u in undefined_symbols:
        undefined_symbols.remove(u)

    x = sympy.MatrixSymbol('x', 3, 1)
    if x in undefined_symbols:
        body.append('x = self.mesh.coords[k]')
        undefined_symbols.remove(x)
        if vertex in unused_arguments:
            unused_arguments.remove(vertex)

    k = sympy.Symbol('k')
    if k in undefined_symbols:
        undefined_symbols.remove(k)
        if vertex in unused_arguments:
            unused_arguments.remove(vertex)

    if len(undefined_symbols) > 0:
        raise RuntimeError(
                'The following symbols are undefined: %s' % undefined_symbols
                )

    # remove double lines
    body = list_unique(body)

    return body
