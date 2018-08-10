# -*- coding: utf-8 -*-
#
from __future__ import print_function

from . import fvm_problem
from . import linear_fvm_problem

from .discretize_linear import discretize_linear, split
from .discretize import discretize
from .nonlinear_methods import newton
from .fvm_matrix import get_fvm_matrix

from pyfvm.__about__ import __version__, __author__, __author_email__

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

try:
    import pipdate
except ImportError:
    pass
else:
    if pipdate.needs_checking(__name__):
        print(pipdate.check(__name__, __version__), end="")
