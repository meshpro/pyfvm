# pyfvm

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/pyfvm/ci?style=flat-square)](https://github.com/nschloe/pyfvm/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/pyfvm.svg?style=flat-square)](https://codecov.io/gh/nschloe/pyfvm)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pyfvm.svg?style=flat-square)](https://pypi.org/pypi/pyfvm/)
[![PyPi Version](https://img.shields.io/pypi/v/pyfvm.svg?style=flat-square)](https://pypi.org/project/pyfvm)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/pyfvm.svg?style=flat-square&logo=github&label=Stars)](https://github.com/nschloe/pyfvm)
[![PyPi downloads](https://img.shields.io/pypi/dm/pyfvm.svg?style=flat-square)](https://pypistats.org/packages/pyfvm)

Creating finite volume equation systems with ease.

pyfvm provides everything that is needed for setting up finite volume equation systems.
The user needs to specify the finite volume formulation in a configuration file, and
pyfvm will create the matrix/right-hand side or the nonlinear system for it. This
package is for everyone who wants to quickly construct FVM systems.

### Examples

#### Linear equation systems

pyfvm works by specifying the residuals, so for solving Poisson's equation with
Dirichlet boundary conditions, simply do
```python,test
import pyfvm
from pyfvm.form_language import *
import meshzoo
from scipy.sparse import linalg
import meshplex

class Poisson(object):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
             - integrate(lambda x: 1.0, dV)

    def dirichlet(self, u):
        return [(lambda x: u(x) - 0.0, Boundary())]

# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle(0.0, 2.0, 0.0, 1.0, 401, 201)
mesh = meshplex.MeshTri(vertices, cells)

matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)

u = linalg.spsolve(matrix, rhs)

mesh.write('out.vtk', point_data={'u': u})
```
This example uses [meshzoo](https://pypi.org/project/meshzoo) for creating a simple
mesh, but anything else that provides vertices and cells works as well. For example,
reading from a wide variety of mesh files is supported (via
[meshio](https://pypi.org/project/meshio)):
```python
mesh = meshplex.read("pacman.e")
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
```python,test
import pyfvm
from pyfvm.form_language import *
import meshzoo
import numpy
from sympy import exp
import meshplex

class Bratu(object):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) \
             - integrate(lambda x: 2.0 * exp(u(x)), dV)

    def dirichlet(self, u):
        return [(u, Boundary())]

vertices, cells = meshzoo.rectangle(0.0, 2.0, 0.0, 1.0, 101, 51)
mesh = meshplex.MeshTri(vertices, cells)

f, jacobian = pyfvm.discretize(Bratu(), mesh)

def jacobian_solver(u0, rhs):
    from scipy.sparse import linalg
    jac = jacobian.get_linear_operator(u0)
    return linalg.spsolve(jac, rhs)

u0 = numpy.zeros(len(vertices))
u = pyfvm.newton(f.eval, jacobian_solver, u0)

mesh.write('out.vtk', point_data={'u': u})
```
Note that the Jacobian is computed symbolically from the `Bratu` class.

Instead of `pyfvm.newton`, you can use any solver that accepts the residual
computation `f.eval`, e.g.,
```
import scipy.optimize
u = scipy.optimize.newton_krylov(f.eval, u0)
```

### Installation

pyfvm is [available from the Python Package
Index](https://pypi.org/project/pyfvm/), so simply type
```
pip install pyfvm
```
to install.

### Testing

To run the tests, check out this repository and type
```
pytest
```

### License
This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
