# PyFVM

[![Build Status](https://travis-ci.org/nschloe/pyfvm.svg?branch=master)](https://travis-ci.org/nschloe/pyfvm)
[![Requirements Status](https://requires.io/github/nschloe/pyfvm/requirements.svg?branch=master)](https://requires.io/github/nschloe/pyfvm/requirements/?branch=master)
[![Code Health](https://landscape.io/github/nschloe/pyfvm/master/landscape.png)](https://landscape.io/github/nschloe/pyfvm/master)
[![codecov](https://codecov.io/gh/nschloe/pyfvm/branch/master/graph/badge.svg)](https://codecov.io/gh/nschloe/pyfvm)
[![PyPi Version](https://img.shields.io/pypi/v/pyfvm.svg)](https://pypi.python.org/pypi/pyfvm)

Creating finite volume equation systems with ease.

PyFVM provides everything that is needed for setting up finite volume equation
systems. The user needs to specify the finite volume formulation in a
configuration file, and PyFVM will create the matrix/right-hand side or the
nonlinear system for it. This package is for everyone who wants to quickly
construct FVM systems.

### Examples

#### Linear equation systems

##### Poisson's equation

For solving Poisson's equation with Dirichlet boundary conditions, simply do
From the configuration file
```python
import pyfvm
from pyfvm.form_language import *
from scipy.sparse import linalg
from sympy import sin, pi


class Poisson(LinearFvmProblem):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
             - integrate(lambda x: 10 * sin(2*pi*x[0]), dV)

    def dirichlet(self, u):
        return [
            (lambda x: u(x) - 0.0, Gamma0()),
            (lambda x: u(x) - 1.0, Gamma1())
            ]

# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle.create_mesh(
        0.0, 2.0,
        0.0, 1.0,
        401, 201,
        zigzag=True
        )
mesh = pyfvm.meshTri.meshTri(vertices, cells)

linear_system = pyfvm.discretize(Poisson(), mesh)

x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

mesh.write('out.vtu', point_data={'x': x})
```
This example uses [meshzoo](https://pypi.python.org/pypi/meshzoo) for creating
a simple mesh, but anything else that provides vertices and cells works as
well. For example, reading from a wide variety of mesh files is supported
(via [meshio](https://pypi.python.org/pypi/meshio)):
```python
mesh, _, _ = pyfvm.reader.read('pacman.e')
```

##### Singular perturbation

#### Nonlinear equation systems

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
    $ make publish
    ```

### License

PyFVM is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
