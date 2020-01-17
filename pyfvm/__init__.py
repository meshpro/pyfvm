# -*- coding: utf-8 -*-
#
from __future__ import print_function

from pyfvm.__about__ import __author__, __author_email__, __version__

from . import fvm_problem, linear_fvm_problem
from .discretize import discretize
from .discretize_linear import discretize_linear, split
from .fvm_matrix import get_fvm_matrix
from .nonlinear_methods import newton

__all__ = [
    "__version__",
    "__author__",
    "__author_email__",
    "discretize",
    "discretize_linear",
    "split",
    "newton",
    "fvm_problem",
    "linear_fvm_problem",
    "get_fvm_matrix",
    "EdgeMatrixKernel",
]
