# -*- coding: utf-8 -*-
#
import inspect
import os
from string import Template
import sympy

from ..fvm_matrix import \
        FvmMatrixCode, \
        gather_core_dependencies
from ..integral_boundary import IntegralBoundary
from ..dirichlet import Dirichlet
from ..integral_edge import IntegralEdge
from ..integral_vertex import IntegralVertex
from ..helpers import get_uuid, sanitize_identifier_cxx


class FvmOperatorCode(object):
    def __init__(self, namespace, cls):
        self.class_name_cxx = sanitize_identifier_cxx(cls.__name__)
        self.namespace = namespace
        self.dependencies = gather_dependencies(namespace, cls)
        return

    def get_dependencies(self):
        return self.dependencies

    def get_cxx_class_object(self, dep_class_objects):
        code = get_code_linear_problem_cxx(
            'fvm_operator.tpl',
            self.class_name_cxx,
            'nosh::fvm_operator',
            self.dependencies
            )

        return {
            'code': code
            }


def gather_dependencies(namespace, cls):
    u = sympy.Function('u')
    u.nosh = True

    u0 = sympy.Function('u0')
    u0.nosh = True

    if (len(inspect.getargspec(cls.apply).args) == 1):
        res = cls.apply(u)
    elif (len(inspect.getargspec(cls.apply).args) == 2):
        res = cls.apply(u, u0)
    else:
        raise ValueError('Only methods with one or two arguments allowed.')

    dependencies = gather_core_dependencies(
            namespace, res, cls.dirichlet, matrix_var=None
            )

    # Add dependencies on fvm_matrices
    for fvm_matrix in res.fvm_matrices:
        dependencies.add(
            FvmMatrixCode(namespace, fvm_matrix.__class__)
            )

    return dependencies


def get_code_linear_problem_cxx(
        template_filename,
        class_name_cxx,
        base_class_name_cxx,
        dependencies
        ):
    dirichlet_cores = []
    vertex_cores = []
    edge_cores = []
    boundary_cores = []
    fvm_matrices = []
    scalar_parameters = []
    vector_parameters = []
    for dep in dependencies:
        # Sort the cores
        if isinstance(dep, Dirichlet):
            id = ('dirichlets_', len(dirichlet_cores))
            dirichlet_cores.append(dep)
        elif isinstance(dep, IntegralVertex):
            id = ('vertex_cores_', len(vertex_cores))
            vertex_cores.append(dep)
        elif isinstance(dep, IntegralEdge):
            id = ('edge_cores_', len(edge_cores))
            edge_cores.append(dep)
        elif isinstance(dep, IntegralBoundary):
            id = ('boundary_cores_', len(boundary_cores))
            boundary_cores.append(dep)
        elif isinstance(dep, FvmMatrixCode):
            id = ('operator_', len(fvm_matrices))
            fvm_matrices.append(dep)
        else:
            raise RuntimeError(
                'Dependency \'%s\' not accounted for.' % dep.class_name_cxx
                )
        # Collect all dependencies
        if dep.scalar_params:
            scalar_parameters.append(id)
        if dep.vector_params:
            vector_parameters.append(id)

    constructor_args = [
        'const std::shared_ptr<const nosh::mesh> & _mesh'
        ]
    init_operator_core_edge = '{%s}' % (
            ', '.join(['std::make_shared<%s>(_mesh)' % n.class_name_cxx
                       for n in edge_cores])
            )
    init_operator_core_vertex = '{%s}' % (
            ', '.join(['std::make_shared<%s>(_mesh)' % n.class_name_cxx
                       for n in vertex_cores])
            )
    init_operator_core_boundary = '{%s}' % (
            ', '.join(['std::make_shared<%s>(_mesh)' % n.class_name_cxx
                       for n in boundary_cores])
            )
    init_operator_core_dirichlet = '{%s}' % (
            ', '.join(['std::make_shared<%s>(_mesh)' % n.class_name_cxx
                       for n in dirichlet_cores])
            )
    init_fvm_matrix = '{%s}' % (
            ', '.join(['std::make_shared<%s>(_mesh)' % n.class_name_cxx
                       for n in fvm_matrices])
            )

    members_init = []
    members_declare = []

    extra_methods = []

    lines_refill = []
    lines_gsp = []
    for id in scalar_parameters:
        core_set, k = id
        var = '%s_scalar_params_%s_' % (core_set, k)
        lines_gsp.append('''const auto %s =
              %s[%s]->get_scalar_parameters();
            scalar_params_map.insert(
              %s.begin(),
              %s.end()
              );''' % (var, core_set, k, var, var)
              )
        lines_refill.append('''
        %s[%s]->refill_(scalar_params, vector_params);
        ''' % (core_set, k))

    lines_gvp = []
    for id in vector_parameters:
        core_set, k = id
        var = '%s_vector_params_%s_' % (core_set, k)
        lines_gvp.append('''const auto %s =
              %s[%s]->get_vector_parameters();
            vector_params_map.insert(
              %s.begin(),
              %s.end()
              );''' % (var, core_set, k, var, var)
              )
        lines_refill.append('''
        %s[%s]->refill_(scalar_params, vector_params);
        ''' % (core_set, k))

    if lines_gsp:
        extra_methods.append('''
        virtual
        std::map<std::string, double>
        get_scalar_parameters() const
        {
          std::map<std::string, double> scalar_params_map = {};
          %s
          return scalar_params_map;
        };
        ''' % '\n'.join(lines_gsp)
        )

    tpetra = 'Tpetra::Vector<double, int, int>'
    if lines_gvp:
        extra_methods.append('''
        virtual
        std::map<std::string, std::shared_ptr<const %s>>
        get_vector_parameters() const
        {
          std::map<std::string, std::shared_ptr<const %s>> vector_params_map = {};
          %s
          return vector_params_map;
        };
        ''' % (tpetra, tpetra, '\n'.join(lines_gvp))
        )

    if lines_refill:
        extra_methods.append('''
        virtual
        void
        refill_(
          const std::map<std::string, double> & scalar_params,
          const std::map<std::string, std::shared_ptr<const %s>> & vector_params
          )
          {%s}
        ''' % (tpetra, '\n'.join(lines_refill))
        )

    with open(template_filename, 'r') as f:
        src = Template(f.read())
        code = src.substitute({
            'name': class_name_cxx,
            'constructor_args': ',\n'.join(constructor_args),
            'members_init': ',\n'.join(members_init),
            'members_declare': '\n'.join(members_declare),
            'extra_methods': '\n'.join(extra_methods),
            'init_edge_cores': init_operator_core_edge,
            'init_vertex_cores': init_operator_core_vertex,
            'init_boundary_cores': init_operator_core_boundary,
            'init_dirichlets': init_operator_core_dirichlet,
            'init_operators': init_fvm_matrix,
            })

    return code
