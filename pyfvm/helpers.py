# -*- coding: utf-8 -*-
#
import sympy


def is_affine_linear(expr, variables):
    for var in variables:
        if not sympy.Eq(sympy.diff(expr, var, var), 0):
            return False
    return True


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
