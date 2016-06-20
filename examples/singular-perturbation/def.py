# -*- coding: utf-8 -*-
from pyfvm.compiler.form_language import *


class Singular(LinearFvmProblem):
    @staticmethod
    def apply(u):
        return integrate(lambda x: - 0.01 * n_dot_grad(u(x)), dS) \
               + integrate(lambda x: u(x), dV) \
               - integrate(lambda x: 1.0, dV)
    dirichlet = [(lambda x: 0.0, Boundary)]
