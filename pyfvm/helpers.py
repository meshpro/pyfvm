# -*- coding: utf-8 -*-
#
import sympy


def split_affine_linear_nonlinear(expr, variables):
    if isinstance(expr, float):
        return expr, 0, 0

    input_is_list = True
    if not isinstance(variables, list):
        input_is_list = False
        variables = [variables]

    # See <https://github.com/sympy/sympy/issues/11475> on why we need expand()
    # here.
    affine = expr.expand()
    linear = []
    for var in variables:
        linear.append(sympy.diff(expr, var).coeff(var, 0))
        affine = affine.coeff(var, 0)

    # The rest is nonlinear
    nonlinear = expr - affine
    for var, coeff in zip(variables, linear):
        nonlinear -= var * coeff
    nonlinear = sympy.simplify(nonlinear)

    if not input_is_list:
        assert len(linear) == 1
        linear = linear[0]

    return affine, linear, nonlinear
