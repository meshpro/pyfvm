# -*- coding: utf-8 -*-
#
import os
import re
import subprocess
import sympy
import sys
import uuid

import nfl

templates_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    'templates'
    )


def get_uuid():
    return str(uuid.uuid4())[:8]


def extract_c_expression(expr):
    from sympy.utilities.codegen import codegen
    # The incoming expression may contain IndexedBase objects which cannot be
    # translated into C code directly, cf.
    # <https://groups.google.com/forum/?utm_medium=email&utm_source=footer#!msg/sympy/VtxujWe5Fkc/Psuy2XHbAwAJ>.
    # Hence, replace all occurrences of IndexedBase objects by simply Symbols
    # with names like 'u[k]'.
    k = sympy.Symbol('k')
    try:
        for s in expr.atoms(sympy.IndexedBase):
            expr = expr.subs(s[k], sympy.Symbol('%s[k]' % s))
    except AttributeError:
        # AttributeError: 'float' object has no attribute 'atoms'
        pass

    [(c_name, c_code), (h_name, c_header)] = codegen(("f", expr), "C")
    res = re.search("f_result = (.*);", c_code)
    return res.group(1)


def run(command):
    """Runs a given command on the command line and returns its output.
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True
        )
    output = process.stdout.read()[:-1]
    ret = process.wait()

    if ret != 0:
        sys.exit(
            "\nERROR: The command \n\n%s\n\nreturned a nonzero "
            "exit status. The error message is \n\n%s\n\n"
            "Abort.\n"
            % (command, process.stderr.read()[:-1])
            )
    return output


def is_affine_linear(expr, vars):
    for var in vars:
        if not sympy.Eq(sympy.diff(expr, var, var), 0):
            return False
    return True


# We still need this for pure matrices
# def is_linear(expr, vars):
#     if not _is_affine_linear(expr, vars):
#         return False
#     # Check that expr is not affine.
#     if isinstance(expr, int) or isinstance(expr, float):
#         return expr == 0
#     else:
#         return expr.subs([(var, 0) for var in vars]) == 0


def compare_variables(arguments, expressions):
    used_symbols = set([])
    used_expressions = set([])

    for expr in expressions:
        try:
            used_symbols.update(expr.free_symbols)
            used_expressions.update(set([
                    type(atom) for atom in expr.atoms(nfl.Expression)
                    ]))
        except AttributeError:
            pass

    unused_arguments = arguments - used_symbols
    undefined_symbols = used_symbols - arguments
    if len(undefined_symbols) > 0:
        raise RuntimeError(
                'The following symbols are undefined: %s' % undefined_symbols
                )

    return unused_arguments, used_expressions


def extract_linear_components(expr, u0):
    assert(is_affine_linear(expr, [u0]))
    # Get coefficient of u0
    coeff = sympy.diff(expr, u0)
    # Get affine part
    if isinstance(expr, float):
        affine = expr
    else:
        affine = expr.subs(u0, 0)
    return coeff, affine


def members_init_declare(namespace, parent_name, dependency_class_objects):
    # now take care of the template substitution
    members_init = []
    members_declare = []
    subdomain_class_names = []
    for dep in dependency_class_objects:
        if dep['type'] == 'subdomain':
            subdomain_class_names.append(dep['class_name'])
        else:
            var_name = dep['class_name']
            members_init.append(
                '%s(%s::%s(%s))' %
                (var_name, namespace,
                 dep['class_name'], ', '.join(dep['constructor_args']))
                )
            members_declare.append(
                'const %s::%s %s;' %
                (namespace, dep['class_name'], var_name)
                )

    if len(subdomain_class_names) == 0:
        subdomain_class_names.append('everywhere')
    # initialize the parent class first
    members_init.insert(
        0,
        'nosh::%s({%s})' %
        (parent_name, ', '.join(['"%s"' % s for s in subdomain_class_names]))
        )
    return members_init, members_declare


def sanitize_identifier(string):
    # turn any string into a valid C++ variable identifier
    return re.sub('\W|^(?=\d)', '_', string).lower()


def list_unique(seq):
    '''http://stackoverflow.com/a/480227/353337
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def replace_nosh_functions(expr):
    fks = []
    if isinstance(expr, float) or isinstance(expr, int):
        pass
    else:
        function_vars = []
        for f in expr.atoms(sympy.Function):
            if hasattr(f, 'nosh'):
                function_vars.append(f)

        for function_var in function_vars:
            # Replace all occurences of u(x) by u[k] (the value at the control
            # volume center)
            f = sympy.IndexedBase('%s' % function_var.func)
            k = sympy.Symbol('k')
            try:
                expr = expr.subs(function_var, f[k])
            except AttributeError:
                # 'int' object has no attribute 'subs'
                pass
            fks.append(f[k])

    return expr, fks
