# -*- coding: utf-8 -*-
from pyfvm.compiler.form_language import *
from sympy import sin


class Gamma0(Subdomain):
    def is_inside(self, x): return x[1] < 0
    is_boundary_only = True


class Gamma1(Subdomain):
    def is_inside(self, x): return x[1] >= 0
    is_boundary_only = True


class Poisson(LinearFvmProblem):
    @staticmethod
    def apply(u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
                - integrate(lambda x: 10 * sin(10*x[0]), dV)

    dirichlet = [
            (lambda x: 0.0, Gamma0),
            (lambda x: 1.0, Gamma1)
            ]
