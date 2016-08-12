# PyFVM

[![Build Status](https://travis-ci.org/nschloe/pyfvm.svg?branch=master)](https://travis-ci.org/nschloe/pyfvm)
[![Code Health](https://landscape.io/github/nschloe/pyfvm/master/landscape.png)](https://landscape.io/github/nschloe/pyfvm/master)
[![codecov](https://codecov.io/gh/nschloe/pyfvm/branch/master/graph/badge.svg)](https://codecov.io/gh/nschloe/pyfvm)
[![PyPi Version](https://img.shields.io/pypi/v/pyfvm.svg)](https://pypi.python.org/pypi/pyfvm)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/pyfvm.svg?style=social&label=Star&maxAge=2592000)](https://github.com/nschloe/pyfvm)

Creating finite volume equation systems with ease.

PyFVM provides everything that is needed for setting up finite volume equation
systems. The user needs to specify the finite volume formulation in a
configuration file, and PyFVM will create the matrix/right-hand side or the
nonlinear system for it. This package is for everyone who wants to quickly
construct FVM systems.

### Examples

#### Linear equation systems

PyFVM works by specifying the residuals, so for solving Poisson's equation with
Dirichlet boundary conditions, simply do
```python
import pyfvm
from pyfvm.form_language import *
import meshzoo
from scipy.sparse import linalg


class Poisson(FvmProblem):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
             - integrate(lambda x: 1.0, dV)

    def dirichlet(self, u):
        return [(lambda x: u(x) - 0.0, 'boundary')]

# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle.create_mesh(0.0, 2.0, 0.0, 1.0, 401, 201)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

linear_system = pyfvm.discretize_linear(Poisson(), mesh)

u = linalg.spsolve(linear_system.matrix, linear_system.rhs)

mesh.write('out.vtu', point_data={'u': u})
```
This example uses [meshzoo](https://pypi.python.org/pypi/meshzoo) for creating
a simple mesh, but anything else that provides vertices and cells works as
well. For example, reading from a wide variety of mesh files is supported
(via [meshio](https://pypi.python.org/pypi/meshio)):
```python
mesh, _, _ = pyfvm.reader.read('pacman.e')
```
Likewise, [PyAMG](https://github.com/pyamg/pyamg) is a much faster solver
for this problem
```
import pyamg
ml = pyamg.smoothed_aggregation_solver(linear_system.matrix)
u = ml.solve(linear_system.rhs, tol=1e-10)
```

More examples are contained in the [examples directory](examples/).

#### Nonlinear equation systems
Nonlinear systems are treated almost equally; only the discretization and
obviously the solver call is different. For Bratu's problem:
```python
import pyfvm
from pyfvm.form_language import *
import meshzoo
import numpy
from sympy import exp


class Bratu(FvmProblem):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
             - integrate(lambda x: 2.0 * exp(u(x)), dV)

    def dirichlet(self, u):
        return [(u, 'boundary')]

vertices, cells = meshzoo.rectangle.create_mesh(0.0, 2.0, 0.0, 1.0, 101, 51)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

f, jacobian = pyfvm.discretize(Bratu(), mesh)

u0 = numpy.zeros(len(vertices))
u = pyfvm.newton(f.eval, jacobian.get_matrix, u0)

mesh.write('out.vtu', point_data={'u': u})
```
Note that the Jacobian is computed symbolically from the `Bratu` class.

Instead of `pyfvm.newton`, you can use any solver that accepts the residual
computation `f.eval`, e.g.,
```
import scipy.optimize
u = scipy.optimize.newton_krylov(f.eval, u0)
```

### Installation

#### Python Package Index

PyFVM is [available from the Python Package
Index](https://pypi.python.org/pypi/pyfvm/), so simply type
```
pip install -U pyfvm
```
to install or upgrade.

#### Manual installation

Download PyFVM from
[the Python Package Index](https://pypi.python.org/pypi/pyfvm/).
Place PyFVM in a directory where Python can find it (e.g.,
`$PYTHONPATH`).  You can install it system-wide with
```
python setup.py install
```

### Distribution

To create a new release

1. bump the `__version__` number,

2. publish to PyPi and GitHub:
    ```
    make publish
    ```

### License

PyFVM is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
