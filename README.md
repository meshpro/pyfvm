# PyFVM

[![Build Status](https://travis-ci.org/nschloe/pyfvm.svg?branch=master)](https://travis-ci.org/nschloe/pyfvm)
[![Requirements Status](https://requires.io/github/nschloe/pyfvm/requirements.svg?branch=master)](https://requires.io/github/nschloe/pyfvm/requirements/?branch=master)
[![Code Health](https://landscape.io/github/nschloe/pyfvm/master/landscape.png)](https://landscape.io/github/nschloe/pyfvm/master)
[![codecov](https://codecov.io/gh/nschloe/pyfvm/branch/master/graph/badge.svg)](https://codecov.io/gh/nschloe/pyfvm)
[![PyPi Version](https://img.shields.io/pypi/v/pyfvm.svg)](https://pypi.python.org/pypi/pyfvm)
[![PyPi Downloads](https://img.shields.io/pypi/dm/pyfvm.svg)](https://pypi.python.org/pypi/pyfvm)

Creating finite volume equation systems with ease.

PyFVM provides everything that is needed for setting up finite volume equation
systems. The user needs to specify the finite volume formulation in a
configuration file, and PyFVM will create the matrix/right-hand side or the
nonlinear system for it. This package is for everyone who wants to quickly
construct FVM systems.

### Examples

#### Linear equation systems

##### Poisson's equation

From the configuration file
```python
from nfl import *
from sympy import sin

class Poisson(LinearFvmProblem):
    def apply(u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
                - integrate(lambda x: 10 * sin(10*x[0]), dV)
```
one creates a Python module via
```bash
form-compiler def.py poisson.py
```
This can then used from any Python module, e.g., for solving the equation
system with SciPy's sparse matrix capabilties:
```python
import poisson

import meshzoo
import pyfvm
from scipy.sparse import linalg

# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle.create_mesh(2.0, 1.0, 21, 11, zigzag=True)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

problem = poisson.Poisson(mesh)

x = linalg.spsolve(problem.matrix, problem.rhs)

mesh.write('out.vtu', point_data={'x': x})
```
This example uses [meshzoo](https://pypi.python.org/pypi/meshzoo) for creating
a simple mesh, but reading from a wide variety of mesh files is supported, too
(via [meshio](https://pypi.python.org/pypi/meshio));
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

2. create a Git tag,
    ```
    $ git tag v0.3.1
    $ git push --tags
    ```
    and

3. upload to PyPi:
    ```
    $ make upload
    ```

### License

PyFVM is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
