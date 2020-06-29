from .__about__ import __version__

from . import fvm_problem, linear_fvm_problem
from .discretize import discretize
from .discretize_linear import discretize_linear, split
from .fvm_matrix import get_fvm_matrix
from .nonlinear_methods import newton

__all__ = [
    "__version__",
    "discretize",
    "discretize_linear",
    "split",
    "newton",
    "fvm_problem",
    "linear_fvm_problem",
    "get_fvm_matrix",
    "EdgeMatrixKernel",
]
