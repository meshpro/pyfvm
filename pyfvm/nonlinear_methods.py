# -*- coding: utf-8 -*-
#
import numpy
from scipy.sparse import linalg


def newton(f, jacobian, u0, tol=1.0e-10, max_iter=20, verbose=True):
    u = u0.copy()

    fu = f(u)
    nrm = numpy.linalg.norm(fu)
    if verbose:
        print('||F(u)|| = %e' % nrm)

    k = 0
    while nrm > tol:
        jac = jacobian(u)
        du = linalg.spsolve(jac, -fu)
        u += du
        fu = f(u)
        nrm = numpy.linalg.norm(fu)
        k += 1
        if verbose:
            print('||F(u)|| = %e' % nrm)

        if k >= max_iter:
            break

    return u