import time

import dufte
import matplotlib.pyplot as plt
import meshplex
import meshzoo
import numpy as np
import pyamg
from scipy.sparse import linalg

import pyfvm
from pyfvm.form_language import Boundary, dS, dV, integrate, n_dot_grad


class Poisson:
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS) - integrate(lambda x: 1.0, dV)

    def dirichlet(self, u):
        return [(lambda x: u(x) - 0.0, Boundary())]


# N = [2 ** k for k in range(2, 13)]
N = [k for k in range(50, 501, 50)]
x = []
times = []
for n in N:
    print(n)
    t0 = time.time()

    vertices, cells = meshzoo.rectangle_tri((0.0, 0.0), (1.0, 1.0), n)
    mesh = meshplex.Mesh(vertices, cells)
    x.append(mesh.points.shape[0])

    t1 = time.time()

    matrix, rhs = pyfvm.discretize_linear(Poisson(), mesh)

    t2 = time.time()

    u = linalg.spsolve(matrix, rhs)

    t3 = time.time()

    ml = pyamg.smoothed_aggregation_solver(matrix)
    u = ml.solve(rhs, tol=1e-10)

    t4 = time.time()

    times.append([t1 - t0, t2 - t1, t3 - t2, t4 - t3])


plt.style.use(dufte.style)

times = np.array(times).T
plt.loglog(x, times[0], label="meshgen")
plt.loglog(x, times[1], label="discretize")
plt.loglog(x, times[2], label="spsolve")
plt.loglog(x, times[3], label="mlsolve")
plt.xlabel("num unknowns")
plt.title("Runtime [s]")
dufte.legend()
plt.show()
