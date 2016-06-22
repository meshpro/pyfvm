# -*- coding: utf-8 -*-
#
import inspect
import os
from string import Template
import sympy

from .code_generator_tpetra import get_code_tpetra
from ..helpers import extract_c_expression


def get_code_operator(namespace, name, var):
    assert(callable(var.eval))

    if (len(inspect.getargspec(var.eval).args) == 1):
        code = _get_code_operator_plain(name, var)
    elif (len(inspect.getargspec(var.eval).args) == 2):
        code = _get_code_operator_with_rebuild(name, var)
    else:
        raise ValueError('Only methods with one or two arguments allowed.')

    return code


def _get_code_operator_plain(name, obj):
    u = sympy.Function('u')
    u.nosh = True

    op_code, required_fvm_matrices = get_code_tpetra(obj.eval(u), {'u': 'x'})

    boundary_eval = getattr(obj, 'boundary_eval', None)
    boundary_code = None
    if callable(boundary_eval):
        xk = sympy.Symbol('xData[k]')
        code = extract_c_expression(boundary_eval(xk))
        boundary_code = '''
        for (const auto & vertex: this->mesh_->boundary_vertices) {
          const auto k = this->mesh_->local_index(vertex);
          yData[k] = %s;
        }
        ''' % code

    dependencies = set()

    light_constructor_args = ['const std::shared_ptr<const nosh::mesh> & mesh']
    full_constructor_args = ['const std::shared_ptr<const nosh::mesh> & mesh']

    light_members_init = ['mesh_(mesh)']
    full_members_init = ['mesh_(mesh)']

    light_members_declare = ['const std::shared_ptr<const nosh::mesh> mesh_;']
    full_members_declare = ['const std::shared_ptr<const nosh::mesh> mesh_;']
    full_members_names = ['mesh_']

    for required_fvm_matrix in required_fvm_matrices:
        var_name = required_fvm_matrix['var_name'].lower()
        arg_name = var_name + '_'
        class_name_cxx = required_fvm_matrix['class'].__name__.lower()
        light_constructor_args.append(
                'const std::shared_ptr<const Tpetra::Operator> & %s' % arg_name
                )
        light_members_init.append(
                '%s(%s)' % (var_name, arg_name)
                )
        # At this point, we actually need to know the dependencies of the
        # class_name_cxx constructor. As a stopgap measure, just put 'mesh' and
        # hope for the best.
        # TODO fix this
        full_members_init.append(
                '%s(std::make_shared<%s>(mesh))' % (var_name, class_name_cxx)
                )
        light_members_declare.append(
                'const std::shared_ptr<const Tpetra::Operator> %s;' % var_name
                )
        full_members_declare.append(
                'const std::shared_ptr<const Tpetra::Operator> %s;' % var_name
                )
        full_members_names.append(var_name)

        dependencies.add(required_fvm_matrix['class'])

    light_class_name_cxx = name.lower() + '_light'
    # initialize a member if _light in the full class
    light_var_name = name.lower() + '_light_'
    full_members_declare.append(
        'const std::shared_ptr<const Tpetra::Operator> %s;' % light_var_name
        )
    full_members_init.append(
        '%s(std::make_shared<%s>(%s)' % (
            light_var_name, light_class_name_cxx, ', '.join(full_members_names)
        )
        )

    # TODO
    # Check if any of the arguments in `apply` is not used in the function.
    # (We'll declare them (void) to supress compiler warnings.)

    if light_members_init:
        light_members_init = ':\n' + ',\n'.join(light_members_init)
    else:
        light_members_init = ''

    if full_members_init:
        full_members_init = ':\n' + ',\n'.join(full_members_init)
    else:
        full_members_init = ''

    with open('operator.tpl', 'r') as f:
        src = Template(f.read())
        code = src.substitute({
            'light_class_name_cxx': name.lower() + '_light',
            'light_constructor_args': ',\n'.join(light_constructor_args),
            'light_members_init': light_members_init,
            'light_members_declare': '\n'.join(light_members_declare),
            'light_apply': op_code,
            'boundary_code': boundary_code,
            'full_class_name_cxx': name.lower(),
            'full_members_init': full_members_init,
            'light_var_name': light_var_name,
            'full_members_declare': '\n'.join(full_members_declare)
            })
    return code, dependencies


def _get_code_operator_with_rebuild():
    code = ''
    generator = CodeGen()
    u = sympy.Function('x')
    u.nosh = True

    u0 = sympy.Function('x0_')
    u0.nosh = True

    op_code, required_fvm_matrices = \
        generator.generate(operator.eval(u, u0))

    members = [
            'const std::shared_ptr<const nosh::mesh> mesh_;'
            'Tpetra::Vector<double,int,int> x0_;'
            ]
    members_init = ['mesh_(mesh)', 'x0_(x0)']
    for required_fvm_matrix in required_fvm_matrices:
        members.append(
                'const std::shared<const Tpetra::Operator> %s;' %
                required_fvm_matrix['var_name'].lower()
                )
        members_init.append(
                '%s(std::make_shared<%s>(mesh))' %
                (required_fvm_matrix['var_name'].lower(),
                    required_fvm_matrix['class'].__name__.lower())
                )
    with open('operator_with_rebuild.tpl', 'r') as f:
        src = Template(f.read())
        code = src.substitute({
            'name': name.lower(),  # class names are lowercase
            'apply': op_code,
            'members': '\n'.join(members),
            'members_init': ',\n'.join(members_init)
            })
    return
