# -*- coding: utf-8 -*-
#
import os
from string import Template
import sympy

from ..expression import *
from ..subdomain import *
from pyfvm.form_language import Expression, ScalarParameter
from ..helpers import \
        extract_c_expression, \
        extract_linear_components, \
        get_uuid, \
        list_unique, \
        cxx_members_init_declare, \
        replace_nosh_functions


class IntegralVertex(object):
    def __init__(self, namespace, integrand, subdomains, matrix_var=None):
        self.namespace = namespace
        uuid = get_uuid()
        self.class_name_cxx = 'vertex_core_' + uuid
        self.class_name_python = 'VertexCore' + uuid

        self.matrix_var = matrix_var

        x = sympy.MatrixSymbol('x', 3, 1)
        fx = integrand(x)
        self.expr, self.vector_vars = _discretize_expression(fx)

        # collect vector parameters
        self.vector_params = set()
        for s in self.expr.atoms(sympy.IndexedBase):
            # `u` is an argument to the kernel and hence already defined
            if s != sympy.IndexedBase('u'):
                self.vector_params.add(s)

        # collect scalar parameters
        self.scalar_params = self.expr.atoms(ScalarParameter)

        self.dependencies = set().union(
            [ExpressionCode(type(atom))
                for atom in self.expr.atoms(Expression)],
            [SubdomainCode(sd) for sd in subdomains]
            )

        self.subdomains = subdomains
        return

    def get_dependencies(self):
        return self.dependencies

    def get_cxx_class_object(self, dependency_class_objects):
        if self.matrix_var:
            arguments = set([sympy.Symbol('vertex')])
        else:
            arguments = set([sympy.Symbol('vertex'), sympy.Symbol('u')])
        used_vars = self.expr.free_symbols

        eval_body = []
        init = []
        declare = []
        methods = []

        # now take care of the template substitution
        deps_init, deps_declare = \
            cxx_members_init_declare(
                    self.namespace,
                    'matrix_core_vertex' if self.matrix_var else
                    'operator_core_vertex',
                    dependency_class_objects
                    )
        init.extend(deps_init)
        declare.extend(deps_declare)

        arguments.update(self.scalar_params)
        # Unfortunately, we cannot just add the vector_params to the arguments
        # since in the used_variables, given by expr.free_symbols, they are
        # reported as sympy.Symbol, not sympy.IndexedBase.
        for v in self.vector_params:
            arguments.add(sympy.Symbol('%s' % v))

        # handle parameters
        params_init, params_declare, params_methods = \
            _handle_parameters_cxx(self.scalar_params, self.vector_params)
        init.extend(params_init)
        declare.extend(params_declare)
        methods.extend(params_methods)

        extra_body, extra_init, extra_declare = _get_cxx_extra(
                arguments, used_vars
                )
        eval_body.extend(extra_body)
        init.extend(extra_init)
        declare.extend(extra_declare)

        # remove double lines
        eval_body = list_unique(eval_body)
        init = list_unique(init)
        declare = list_unique(declare)

        if self.matrix_var:
            coeff, affine = extract_linear_components(
                    self.expr,
                    sympy.Symbol('%s[k]' % self.matrix_var)
                    )
            obj_type = 'matrix_core_vertex'
            filename = os.path.join(
                    os.path.dirname(__file__),
                    'cxx_matrix_core_vertex.tpl'
                    )
            with open(filename, 'r') as f:
                src = Template(f.read())
                code = src.substitute({
                    'name': self.class_name_cxx,
                    'vertex_contrib': extract_c_expression(coeff),
                    'vertex_affine': extract_c_expression(-affine),
                    'vertex_body': '\n'.join(eval_body),
                    'members_init': ':\n' + ',\n'.join(init) if init else '',
                    'members_declare': '\n'.join(declare)
                    })
        else:
            obj_type = 'operator_core_vertex'
            filename = os.path.join(
                    os.path.dirname(__file__),
                    'cxx_operator_core_vertex.tpl'
                    )
            with open(filename, 'r') as f:
                src = Template(f.read())
                code = src.substitute({
                    'name': self.class_name_cxx,
                    'return_value': extract_c_expression(self.expr),
                    'eval_body': '\n'.join(eval_body),
                    'members_init': ':\n' + ',\n'.join(init) if init else '',
                    'members_declare': '\n'.join(declare),
                    'methods': '\n'.join(methods)
                    })

        return {
            'type': obj_type,
            'code': code,
            'class_name_cxx': self.class_name_cxx,
            'constructor_args': [],
            'scalar_parameters': self.scalar_params,
            'vector_parameters': self.vector_params
            }

    def get_python_class_object(self, dependency_class_objects):
        if self.matrix_var:
            arguments = set([sympy.Symbol('k')])
        else:
            arguments = set([sympy.Symbol('k'), sympy.Symbol('u')])

        eval_body = []
        init = []
        declare = []
        methods = []

        arguments.update(self.scalar_params)
        # Unfortunately, we cannot just add the vector_params to the arguments
        # since in the used_variables, given by expr.free_symbols, they are
        # reported as sympy.Symbol, not sympy.IndexedBase.
        for v in self.vector_params:
            arguments.add(sympy.Symbol('%s' % v))

        # handle parameters
        params_init, params_declare, params_methods = \
            _handle_parameters_cxx(self.scalar_params, self.vector_params)
        init.extend(params_init)
        declare.extend(params_declare)
        methods.extend(params_methods)

        # collect subdomain init code
        if self.subdomains:
            init_subdomains = '[%s]' % ', '.join(
                    '\'%s\'' % sd.__name__
                    for sd in self.subdomains
                    )
        else:
            init_subdomains = '[\'everywhere\']'
        init.append('self.subdomains = %s ' % init_subdomains)

        # remove double lines
        eval_body = list_unique(eval_body)
        init = list_unique(init)
        declare = list_unique(declare)

        if self.matrix_var:
            # Unfortunately, it's not too easy to differentiate with respect to
            # an IndexedBase u with index k. For this reason, we'll simply
            # replace u[k] by a variable uk0.
            u = sympy.IndexedBase('%s' % self.matrix_var)
            k0 = sympy.Symbol('k')
            uk0 = sympy.Symbol('uk0')
            expr = self.expr.subs([(u[k0], uk0)])
            coeff, affine = extract_linear_components(expr, uk0)
            used_vars = coeff.free_symbols.union(affine.free_symbols)
            extra_body = _get_python_extra(
                    arguments, used_vars
                    )
            eval_body.extend(extra_body)
            obj_type = 'matrix_core_vertex'
            filename = os.path.join(
                    os.path.dirname(__file__),
                    'python_matrix_core_vertex.tpl'
                    )
            with open(filename, 'r') as f:
                src = Template(f.read())
                code = src.substitute({
                    'name': self.class_name_python,
                    'vertex_contrib': extract_c_expression(coeff),
                    'vertex_affine': extract_c_expression(-affine),
                    'vertex_body': '; '.join(eval_body),
                    'members_init': ',\n'.join(init),
                    'members_declare': '\n'.join(declare)
                    })
        else:
            obj_type = 'operator_core_vertex'
            filename = os.path.join(
                    os.path.dirname(__file__),
                    'python_operator_core_vertex.tpl'
                    )
            with open(filename, 'r') as f:
                src = Template(f.read())
                code = src.substitute({
                    'name': self.class_name_cxx,
                    'return_value': extract_c_expression(self.expr),
                    'eval_body': '\n'.join(eval_body),
                    'members_init': ':\n' + ',\n'.join(init) if init else '',
                    'members_declare': '\n'.join(declare),
                    'methods': '\n'.join(methods)
                    })

        return {
            'type': obj_type,
            'code': code,
            'class_name_cxx': self.class_name_cxx,
            'constructor_args': [],
            'scalar_parameters': self.scalar_params,
            'vector_parameters': self.vector_params
            }

# def get_matrix_core_vertex_code(namespace, class_name_cxx, core):
#     '''Get code generator from raw core object.
#     '''
#     # handle the vertex contributions
#     x = sympy.MatrixSymbol('x')
#     vol = sympy.Symbol('control_volume')
#     all_symbols = set([x, vol])
#
#     specs = inspect.getargspec(method)
#     assert(len(specs.args) == len(all_symbols) + 1)
#
#     vertex_coeff, vertex_affine = method(x, vol)
#
#     return _get_code_matrix_core_vertex(
#             namespace, class_name_cxx,
#             vertex_coeff, vertex_affine
#             )


def _discretize_expression(expr):
    expr, fks = replace_nosh_functions(expr)
    return sympy.Symbol('control_volume') * expr, fks


def _get_cxx_extra(arguments, used_variables):
    vertex = sympy.Symbol('vertex')
    unused_arguments = arguments - used_variables
    undefined_symbols = used_variables - arguments

    init = []
    body = []
    declare = []

    control_volume = sympy.Symbol('control_volume')
    if control_volume in undefined_symbols:
        init.append('mesh_(mesh)')
        declare.append('const std::shared_ptr<const nosh::mesh> mesh_;')
        init.append('c_data_(mesh->control_volumes()->getData())')
        declare.append('const Teuchos::ArrayRCP<const double> c_data_;')
        body.append('const auto k = this->mesh_->local_index(vertex);')
        body.append('const auto control_volume = this->c_data_[k];')
        undefined_symbols.remove(control_volume)
        if vertex in unused_arguments:
            unused_arguments.remove(vertex)

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

    for name in unused_arguments:
        body.insert(0, '(void) %s;' % name)

    return body, init, declare


def _handle_parameters_cxx(scalar_params, vector_params):
    '''Treat vector variables (u, u0,...)
    '''
    params_init = []
    params_declare = []
    params_methods = []

    tpetra_str = 'Tpetra::Vector<double, int, int>'
    for v in vector_params:
        params_init.extend([
            'mesh_(mesh)',
            '%s_vec_(std::make_shared<%s>(Teuchos::rcp(mesh->map())))'
            % (v, tpetra_str),
            '%s(%s_vec_->getData())' % (v, v)
            ])
        params_declare.extend([
            'const std::shared_ptr<const nosh::mesh> mesh_;',
            'std::shared_ptr<const %s> %s_vec_;' % (tpetra_str, v),
            'Teuchos::ArrayRCP<const double> %s;' % v
            ])

    for alpha in scalar_params:
        params_init.append('%s(0.0)' % alpha)
        params_declare.append('double %s;' % alpha)

    refill_body = []

    if len(vector_params) > 0:
        params_methods.append('''
        virtual
        std::map<std::string, std::shared_ptr<const %s>>
        get_vector_parameters() const
        {
          return {
            %s
            };
        };
        ''' % (
            tpetra_str,
            ',\n'.join(['{"%s", %s_vec_}' % (v, v) for v in vector_params])
            )
        )
        refill_body.append(
          ',\n'.join(['''
          this->%s_vec_ = vector_params.at("%s");
          this->%s = this->%s_vec_->getData();
          ''' % (vec, vec, vec, vec) for vec in vector_params])
          )

    if len(scalar_params) > 0:
        params_methods.append('''
        virtual
        std::map<std::string, double>
        get_scalar_parameters() const
        {
          return {
            %s
            };
        };
        ''' % (
            ',\n'.join(['{"%s", %s}' % (p, p) for p in scalar_params])
            )
        )
        refill_body.append(
          ',\n'.join(['''
          %s = scalar_params.at("%s");
          ''' % (a, a) for a in scalar_params])
          )

    refill_body = list_unique(refill_body)
    if len(refill_body) > 0:
        params_methods.append('''
        virtual
        void
        refill_(
            const std::map<std::string, double> & scalar_params,
            const std::map<std::string, std::shared_ptr<const %s>> & vector_params
            )
        {%s}
        ''' % (
          tpetra_str,
          '\n'.join(refill_body)
        )
        )

    return params_init, params_declare, params_methods


def _get_python_extra(arguments, used_variables):
    vertex = sympy.Symbol('vertex')
    unused_arguments = arguments - used_variables
    undefined_symbols = used_variables - arguments

    body = []

    control_volume = sympy.Symbol('control_volume')
    if control_volume in undefined_symbols:
        body.append('control_volume = self.mesh.control_volumes[k]')
        undefined_symbols.remove(control_volume)
        if vertex in unused_arguments:
            unused_arguments.remove(vertex)

    x = sympy.MatrixSymbol('x', 3, 1)
    if x in undefined_symbols:
        body.append('x = self.mesh.node_coords[k]')
        undefined_symbols.remove(x)
        if vertex in unused_arguments:
            unused_arguments.remove(vertex)

    if len(undefined_symbols) > 0:
        raise RuntimeError(
                'The following symbols are undefined: %s' % undefined_symbols
                )

    return body


def _handle_parameters_cxx(scalar_params, vector_params):
    '''Treat vector variables (u, u0,...)
    '''
    params_init = []
    params_declare = []
    params_methods = []

    tpetra_str = 'Tpetra::Vector<double, int, int>'
    for v in vector_params:
        params_init.extend([
            'mesh_(mesh)',
            '%s_vec_(std::make_shared<%s>(Teuchos::rcp(mesh->map())))' % (v, tpetra_str),
            '%s(%s_vec_->getData())' % (v, v)
            ])
        params_declare.extend([
            'const std::shared_ptr<const nosh::mesh> mesh_;',
            'std::shared_ptr<const %s> %s_vec_;' % (tpetra_str, v),
            'Teuchos::ArrayRCP<const double> %s;' % v
            ])

    for alpha in scalar_params:
        params_init.append('%s(0.0)' % alpha)
        params_declare.append('double %s;' % alpha)

    refill_body = []

    if len(vector_params) > 0:
        params_methods.append('''
        virtual
        std::map<std::string, std::shared_ptr<const %s>>
        get_vector_parameters() const
        {
          return {
            %s
            };
        };
        ''' % (
            tpetra_str,
            ',\n'.join(['{"%s", %s_vec_}' % (v, v) for v in vector_params])
            )
        )
        refill_body.append(
          ',\n'.join(['''
          this->%s_vec_ = vector_params.at("%s");
          this->%s = this->%s_vec_->getData();
          ''' % (vec, vec, vec, vec) for vec in vector_params])
          )

    if len(scalar_params) > 0:
        params_methods.append('''
        virtual
        std::map<std::string, double>
        get_scalar_parameters() const
        {
          return {
            %s
            };
        };
        ''' % (
            ',\n'.join(['{"%s", %s}' % (p, p) for p in scalar_params])
            )
        )
        refill_body.append(
          ',\n'.join(['''
          %s = scalar_params.at("%s");
          ''' % (a, a) for a in scalar_params])
          )

    refill_body = list_unique(refill_body)
    if len(refill_body) > 0:
        params_methods.append('''
        virtual
        void
        refill_(
            const std::map<std::string, double> & scalar_params,
            const std::map<std::string, std::shared_ptr<const %s>> & vector_params
            )
        {%s}
        ''' % (
          tpetra_str,
          '\n'.join(refill_body)
        )
        )

    return params_init, params_declare, params_methods


def _handle_parameters_python(scalar_params, vector_params):
    '''Treat vector variables (u, u0,...)
    '''
    params_init = []
    params_methods = []

    for v in vector_params:
        params_init.extend([
            'self.%s = None' % v
            ])

    for alpha in scalar_params:
        params_init.append('%s = 0.0' % alpha)

    refill_body = []

    if len(vector_params) > 0:
        params_methods.append('''
        def get_vector_parameters(self):
            return {
              %s
              }
        ''' % (
            ',\n'.join(['\'%s\': %s' % (v, v) for v in vector_params])
            )
        )
        refill_body.append(
          ',\n'.join(['''
          self.%s = vector_params[\'%s\']
          ''' % (vec, vec) for vec in vector_params])
          )

    if len(scalar_params) > 0:
        params_methods.append('''
        def get_scalar_parameters(self):
          return {
            %s
            }
        ''' % (
            ',\n'.join(['\'%s\': %s' % (p, p) for p in scalar_params])
            )
        )
        refill_body.append(
          ',\n'.join(['''
          %s = scalar_params[\'%s\']
          ''' % (a, a) for a in scalar_params])
          )

    refill_body = list_unique(refill_body)
    if len(refill_body) > 0:
        params_methods.append('''
        def refill_(self, scalar_params, vector_params):
            %s
            return
        ''' % '\n'.join(refill_body)
        )

    return params_init, params_methods
