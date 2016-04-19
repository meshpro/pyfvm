# -*- coding: utf-8 -*-
#
import os
from string import Template
import sympy
from .helpers import extract_c_expression, templates_dir


def get_code_dirichlet(name, function, subdomains):
    dependencies = subdomains

    x = sympy.MatrixSymbol('x', 3, 1)
    u = sympy.Function('u')
    u.nosh = True

    arguments = set([x, u])
    result = function(x, u)
    try:
        used_variables = result.free_symbols
    except AttributeError:
        used_variables = set()

    unused_arguments = arguments - used_variables
    undefined_variables = used_variables - arguments
    assert(len(undefined_variables) == 0)

    subdomain_ids = set([
        sd.__class__.__name__.lower() for sd in subdomains
        ])

    if len(subdomain_ids) == 0:
        # If nothing is specified, use the entire boundary
        subdomain_ids.add('boundary')

    init = '{%s}' % ', '.join(['"%s"' % s for s in subdomain_ids])

    # template substitution
    filename = os.path.join(templates_dir, 'operator_core_dirichlet.tpl')
    with open(filename, 'r') as f:
        src = Template(f.read())
        code = src.substitute({
            'name': name.lower(),
            'init': 'nosh::operator_core_dirichlet(%s)' % init,
            'eval_return_value': extract_c_expression(result),
            'eval_body': '\n'.join([
                '(void) %s;' % name for name in unused_arguments
                ]),
            })

    return code, dependencies
