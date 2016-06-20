# -*- coding: utf-8 -*-
from sympy import *
from nfl import *


class Singular(LinearFvmProblem):
    def apply(u):
        return integrate(lambda x: - 0.01 * n_dot_grad(u(x)), dS) \
               + integrate(lambda x: u(x), dV) \
               - integrate(lambda x: 1.0, dV)
    dirichlet = [(lambda x: 0.0, Boundary)]
