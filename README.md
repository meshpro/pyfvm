# pyfvm

[![PyPi Version](https://img.shields.io/pypi/v/pyfvm.svg?style=flat-square)](https://pypi.org/project/pyfvm)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pyfvm.svg?style=flat-square)](https://pypi.org/pypi/pyfvm/)
[![GitHub stars](https://img.shields.io/github/stars/meshpro/pyfvm.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/meshpro/pyfvm)
[![PyPi downloads](https://img.shields.io/pypi/dm/pyfvm.svg?style=flat-square)](https://pypistats.org/packages/pyfvm)

[![Discord](https://img.shields.io/static/v1?logo=discord&logoColor=white&label=chat&message=on%20discord&color=7289da&style=flat-square)](https://discord.gg/hnTJ5MRX2Y)

Creating finite volume equation systems with ease.

pyfvm provides everything that is needed for setting up finite volume equation
systems. The user needs to specify the finite volume formulation in a
configuration file, and pyfvm will create the matrix/right-hand side or the
nonlinear system for it. This package is for everyone who wants to quickly
construct FVM systems.

### Examples

#### Linear equation systems

pyfvm works by specifying the residuals, so for solving Poisson's equation with
Dirichlet boundary conditions, simply do

```python
import meshplex
import meshzoo
import numpy as np
from scipy.sparse import linalg

import pyfvm
from pyfvm.form_language import Boundary, dS, dV, integrate, n_dot_grad


class Poisson:
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) - integrate(lambda x: 1.0, dV)

    def dirichlet(self, u):
        return [(lambda x: u(x) - 0.0, Boundary())]


# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle_tri(
    np.linspace(0.0, 2.0, 401), np.linspace(0.0, 1.0, 201)
)
mesh = meshplex.Mesh(vertices, cells)

matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)

u = linalg.spsolve(matrix, rhs)

mesh.write("out.vtk", point_data={"u": u})
```

This example uses [meshzoo](https://pypi.org/project/meshzoo) for creating a
simple mesh, but anything else that provides vertices and cells works as well.
For example, reading from a wide variety of mesh files is supported (via
[meshio](https://pypi.org/project/meshio)):

<!--pytest-codeblocks:skip-->

```python
mesh = meshplex.read("pacman.e")
```

Likewise, [PyAMG](https://github.com/pyamg/pyamg) is a much faster solver for
this problem

<!--pytest-codeblocks:skip-->

```python
import pyamg

ml = pyamg.smoothed_aggregation_solver(matrix)
u = ml.solve(rhs, tol=1e-10)
```

More examples are contained in the [examples directory](examples/).

#### Nonlinear equation systems

Nonlinear systems are treated almost equally; only the discretization and
obviously the solver call is different. For Bratu's problem:

```python
import pyfvm
from pyfvm.form_language import *
import meshzoo
import numpy as np
from sympy import exp
import meshplex


class Bratu:
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) - integrate(
            lambda x: 2.0 * exp(u(x)), dV
        )

    def dirichlet(self, u):
        return [(u, Boundary())]


vertices, cells = meshzoo.rectangle_tri(
    np.linspace(0.0, 2.0, 101), np.linspace(0.0, 1.0, 51)
)
mesh = meshplex.Mesh(vertices, cells)

f, jacobian = pyfvm.discretize(Bratu(), mesh)


def jacobian_solver(u0, rhs):
    from scipy.sparse import linalg

    jac = jacobian.get_linear_operator(u0)
    return linalg.spsolve(jac, rhs)


u0 = np.zeros(len(vertices))
u = pyfvm.newton(f.eval, jacobian_solver, u0)

mesh.write("out.vtk", point_data={"u": u})
```

Note that the Jacobian is computed symbolically from the `Bratu` class.

Instead of `pyfvm.newton`, you can use any solver that accepts the residual
computation `f.eval`, e.g.,

<!--pytest-codeblocks:skip-->

```python
import scipy.optimize

u = scipy.optimize.newton_krylov(f.eval, u0)
```
