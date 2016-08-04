# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy
import pyamg
import pyfvm
import sympy
from scipy.sparse import linalg


def perform_convergence_tests(
        problem, exact_sol, get_mesh, rng, verbose=False
        ):
    n = len(rng)
    H = numpy.empty(n)
    error_norm_1 = numpy.empty(n)
    order_1 = numpy.empty(n-1)
    error_norm_inf = numpy.empty(n)
    order_inf = numpy.empty(n-1)

    if verbose:
        print(79 * '-')
        print('k' + 5*' ' + 'num verts' + 4*' ' + 'max edge length' + 4*' ' +
              '||error||_1' + 8*' ' + '||error||_inf'
              )
        print(38*' ' + '(order)' + 12*' ' + '(order)')
        print(79 * '-')

    # Add "zero" to all entities. This later gets translated into
    # np.zeros with the appropriate length, making sure that scalar
    # terms in the lambda expression correctly return np.arrays.
    zero = sympy.Symbol('zero')
    x = sympy.DeferredVector('x')
    # See <http://docs.sympy.org/dev/modules/utilities/lambdify.html>.
    array2array = [{'ImmutableMatrix': numpy.array}, 'numpy']
    exact_eval = sympy.lambdify((x, zero), exact_sol(x), modules=array2array)

    for k in rng:
        mesh = get_mesh(k)
        H[k] = max(mesh.edge_lengths)

        linear_system = pyfvm.discretize(problem, mesh)

        # x = linalg.spsolve(linear_system.matrix, linear_system.rhs)
        ml = pyamg.ruge_stuben_solver(linear_system.matrix)
        x = ml.solve(linear_system.rhs, tol=1e-10)

        zero = numpy.zeros(len(mesh.node_coords))
        error = x - exact_eval(mesh.node_coords.T, zero)

        # import meshio
        # meshio.write(
        #     'sol%d.vtu' % k,
        #     mesh.node_coords, {'triangle': mesh.cells['nodes']},
        #     point_data={'x': x, 'error': error},
        #     )

        error_norm_1[k] = numpy.sum(abs(mesh.control_volumes * error))
        error_norm_inf[k] = max(abs(error))

        # numerical orders of convergence
        if k > 0:
            order_1[k-1] = \
                numpy.log(error_norm_1[k-1] / error_norm_1[k]) / \
                numpy.log(H[k-1] / H[k])
            order_inf[k-1] = \
                numpy.log(error_norm_inf[k-1] / error_norm_inf[k]) / \
                numpy.log(H[k-1] / H[k])
            if verbose:
                print
                print((38*' ' + '%0.5f' + 12*' ' + '%0.5f') %
                      (order_1[k-1], order_inf[k-1])
                      )
                print

        if verbose:
            num_nodes = len(mesh.control_volumes)
            print('%2d    %5.3e    %0.10e   %0.10e   %0.10e' %
                  (k, num_nodes, H[k], error_norm_1[k], error_norm_inf[k])
                  )

    return H, error_norm_1, error_norm_inf, order_1, order_inf


def plot_error_data(H, error_norm_1, error_norm_inf):
    # plot error data
    plt.loglog(H, error_norm_1, 'xk', label='||error||_1')
    plt.loglog(H, error_norm_inf, 'ok', label='||error||_inf')

    # plot 2nd order indicator
    e0 = max(error_norm_1[0], error_norm_inf[0])
    order = 2
    plt.loglog(
        [H[0], H[-1]],
        [10*e0, 10*e0 * (H[-1]/H[0])**order],
        '--k',
        label='2nd order'
        )

    plt.legend(loc='upper left')
