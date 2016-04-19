# -*- coding: utf-8 -*-
#
import os
from string import Template
import sympy
import nfl

from .discretize_edge_integral import discretize_edge_integral
from ..code_generator_eigen import CodeGeneratorEigen
from ..expression import *
from ..subdomain import *
from ..helpers import \
        extract_c_expression, \
        is_affine_linear, \
        list_unique, \
        get_uuid, \
        members_init_declare


class IntegralEdge(object):
    def __init__(self, namespace, integrand, subdomains, matrix_var=None):
        self.namespace = namespace
        self.class_name = 'edge_core_' + get_uuid()

        self.expr, self.vector_vars = discretize_edge_integral(integrand)

        self.matrix_var = matrix_var

        # collect vector parameters
        self.vector_params = set()
        for s in self.expr.atoms(sympy.IndexedBase):
            # `u` is an argument to the kernel and hence already defined
            if s != sympy.IndexedBase('u'):
                self.vector_params.add(s)

        # gather expressions and subdomains as dependencies
        self.dependencies = set().union(
            [ExpressionCode(type(atom))
                for atom in self.expr.atoms(nfl.Expression)],
            [SubdomainCode(sd) for sd in subdomains]
            )
        return

    def get_dependencies(self):
        return self.dependencies

    def get_cxx_class_object(self, dependency_class_objects):
        init, members_declare = \
            members_init_declare(
                    self.namespace,
                    'matrix_core_edge',
                    dependency_class_objects
                    )

        # Arguments from the template.
        if self.matrix_var:
            type = 'nosh::matrix_core_edge'

            # Unfortunately, it's not too easy to differentiate with respect to
            # an IndexedBase u with indices k0, k1 respectively. For this
            # reason, we'll simply replace u[k0] by a variable uk0, and u[k1]
            # likewise.
            u = sympy.IndexedBase('%s' % self.matrix_var)
            k0 = sympy.Symbol('k0')
            k1 = sympy.Symbol('k1')
            uk0 = sympy.Symbol('uk0')
            uk1 = sympy.Symbol('uk1')
            expr = self.expr.subs([(u[k0], uk0), (u[k1], uk1)])
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

            extra_body, extra_init, extra_declare = \
                _get_extra(arguments, used_vars)

            init.extend(extra_init)
            members_declare.extend(extra_declare)

            # template substitution
            filename = os.path.join(
                    os.path.dirname(__file__),
                    'matrix_core_edge.tpl'
                    )
            with open(filename, 'r') as f:
                src = Template(f.read())
                code = src.substitute({
                    'name': self.class_name,
                    'edge00': _expr_to_code(edge_coeff[0][0]),
                    'edge01': _expr_to_code(edge_coeff[0][1]),
                    'edge10': _expr_to_code(edge_coeff[1][0]),
                    'edge11': _expr_to_code(edge_coeff[1][1]),
                    'edge_affine0': _expr_to_code(-edge_affine[0]),
                    'edge_affine1': _expr_to_code(-edge_affine[1]),
                    'eval_body': '\n'.join(extra_body),
                    'members_init': ':\n' + ',\n'.join(init) if init else '',
                    'members_declare': '\n'.join(members_declare)
                    })
        else:
            used_vars = self.expr.free_symbols

            type = 'nosh::operator_core_edge'
            # template substitution
            filename = os.path.join(
                    os.path.dirname(__file__),
                    'operator_core_edge.tpl'
                    )
            with open(filename, 'r') as f:
                src = Template(f.read())
                code = src.substitute({
                    'name': self.class_name,
                    'edge00': _expr_to_code(edge_coeff[0][0]),
                    'edge01': _expr_to_code(edge_coeff[0][1]),
                    'edge10': _expr_to_code(edge_coeff[1][0]),
                    'edge11': _expr_to_code(edge_coeff[1][1]),
                    'edge_affine0': _expr_to_code(-edge_affine[0]),
                    'edge_affine1': _expr_to_code(-edge_affine[1]),
                    'eval_body': '\n'.join(eval_body),
                    'members_init': ':\n' + ',\n'.join(init) if init else '',
                    'members_declare': '\n'.join(members_declare)
                    })

        return {
            'type': type,
            'code': code,
            'class_name': self.class_name,
            'constructor_args': []
            }


def _extract_linear_components(expr, dvars):
    # Those are the variables in the expression, inserted by the edge
    # discretizer.
    if not is_affine_linear(expr, dvars):
        raise RuntimeError((
            'The given function\n'
            '    f(x) = %s\n'
            'does not seem to be affine linear in u.')
            % function(x)
            )

    # Get the coefficients of u0, u1.
    coeff00 = sympy.diff(expr, dvars[0])
    coeff01 = sympy.diff(expr, dvars[1])

    # Now construct the coefficients for the other way around.
    coeff10 = coeff01.subs([
        (dvars[0], dvars[1]),
        (dvars[1], dvars[0]),
        (nfl.n, nfl.neg_n)
        ])
    coeff11 = coeff00.subs([
        (dvars[0], dvars[1]),
        (dvars[1], dvars[0]),
        (nfl.n, nfl.neg_n)
        ])

    affine = expr.subs([(dvars[0], 0), (dvars[1], 0)])

    return (
        [[coeff00, coeff01], [coeff10, coeff11]],
        [affine, affine]
        )


def _expr_to_code(expr):
    gen = CodeGeneratorEigen()
    code, _ = gen.generate(expr)
    return code


def _get_extra(arguments, used_variables):
    edge = sympy.Symbol('edge')
    unused_arguments = arguments - used_variables
    undefined_symbols = used_variables - arguments

    init = []
    body = []
    declare = []

    edge_length = sympy.Symbol('edge_length')
    if edge_length in undefined_symbols:
        init.append('mesh_(mesh)')
        declare.append('const std::shared_ptr<const nosh::mesh> mesh_;')
        init.append('edge_data_(mesh->get_edge_data())')
        declare.append('const std::vector<nosh::mesh::edge_data> edge_data_;')
        body.append('const auto k = this->mesh_->local_index(edge);')
        body.append('const auto edge_length = this->edge_data_[k].length;')
        undefined_symbols.remove(edge_length)
        if edge in unused_arguments:
            unused_arguments.remove(edge)

    edge_covolume = sympy.Symbol('edge_covolume')
    if edge_covolume in undefined_symbols:
        init.append('mesh_(mesh)')
        declare.append('const std::shared_ptr<const nosh::mesh> mesh_;')
        init.append('edge_data_(mesh->get_edge_data())')
        declare.append('const std::vector<nosh::mesh::edge_data> edge_data_;')
        body.append('const auto k = this->mesh_->local_index(edge);')
        body.append('const auto edge_covolume = this->edge_data_[k].covolume;')
        undefined_symbols.remove(edge_covolume)
        if edge in unused_arguments:
            unused_arguments.remove(edge)

    x0 = sympy.Symbol('x0')
    if x0 in undefined_symbols:
        init.append('mesh_(mesh)')
        declare.append('const std::shared_ptr<const nosh::mesh> mesh_;')
        body.append('const auto verts = this->mesh_->get_vertex_tuple(edge);')
        body.append('const auto x0 = this->mesh_->get_coords(verts[0]);')
        undefined_symbols.remove(x0)
        if edge in unused_arguments:
            unused_arguments.remove(edge)

    x1 = sympy.Symbol('x1')
    if x1 in undefined_symbols:
        init.append('mesh_(mesh)')
        declare.append('const std::shared_ptr<const nosh::mesh> mesh_;')
        body.append('const auto verts = this->mesh_->get_vertex_tuple(edge);')
        body.append('const auto x1 = this->mesh_->get_coords(verts[1]);')
        undefined_symbols.remove(x1)
        if edge in unused_arguments:
            unused_arguments.remove(edge)

    if nfl.n in undefined_symbols:
        init.append('mesh_(mesh)')
        declare.append('const std::shared_ptr<const nosh::mesh> mesh_;')
        body.append('const auto verts = this->mesh_->get_vertex_tuple(edge);')
        body.append('const auto x0 = this->mesh_->get_coords(verts[0]);')
        body.append('const auto x1 = this->mesh_->get_coords(verts[1]);')
        init.append('edge_data_(mesh->get_edge_data())')
        declare.append('const std::vector<nosh::mesh::edge_data> edge_data_;')
        body.append('const auto k = this->mesh_->local_index(edge);')
        body.append('const auto edge_length = this->edge_data_[k].length;')
        body.append('const auto n = (x1 - x0) / edge_length;')
        undefined_symbols.remove(nfl.n)
        if edge in unused_arguments:
            unused_arguments.remove(edge)

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
