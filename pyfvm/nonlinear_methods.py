# -*- coding: utf-8 -*-
#
import numpy


def newton(f, jacobian_solver, u0, tol=1.0e-10, max_iter=20, verbose=True):
    u = u0.copy()

    fu = f(u)
    nrm = numpy.linalg.norm(fu)
    if verbose:
        print('||F(u)|| = %e' % nrm)

    k = 0
    is_converged = False
    while k < max_iter:
        if nrm < tol:
            is_converged = True
            break
        du = jacobian_solver(u, -fu)
        u += du
        fu = f(u)
        nrm = numpy.linalg.norm(fu)
        k += 1
        if verbose:
            print('||F(u)|| = %e' % nrm)

    assert is_converged

    return u
