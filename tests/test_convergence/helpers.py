# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy
import pyfvm
from scipy.sparse import linalg


def perform_convergence_tests(
        problem, exact_sol, get_mesh, rng, do_print=False
        ):
    n = len(rng)
    H = numpy.empty(n)
    error_norm_1 = numpy.empty(n)
    order_1 = numpy.empty(n-1)
    error_norm_inf = numpy.empty(n)
    order_inf = numpy.empty(n-1)

    if do_print:
        print(60 * '-')
        print('k' + 5*' ' + 'h' + 18*' ' +
              '||error||_1' + 8*' ' + '||error||_inf'
              )
        print(' ' + 5*' ' + ' ' + 18*' ' + '(order)' + 12*' ' + '(order)')
        print(60 * '-')

    for k in rng:
        mesh = get_mesh(k)
        H[k] = max(mesh.edge_lengths)

        linear_system = pyfvm.discretize(problem, mesh)

        x = linalg.spsolve(linear_system.matrix, linear_system.rhs)

        diff = x - exact_sol(mesh.node_coords.T)

        error_norm_1[k] = numpy.sum(abs(mesh.control_volumes * diff))
        error_norm_inf[k] = max(abs(diff))

        # numerical orders of convergence
        if k > 0:
            order_1[k-1] = \
                numpy.log(error_norm_1[k-1] / error_norm_1[k]) / \
                numpy.log(H[k-1] / H[k])
            order_inf[k-1] = \
                numpy.log(error_norm_inf[k-1] / error_norm_inf[k]) / \
                numpy.log(H[k-1] / H[k])
            if do_print:
                print
                print((25*' ' + '%0.5f' + 12*' ' + '%0.5f') %
                      (order_1[k-1], order_inf[k-1])
                      )
                print

        if do_print:
            print('%2d    %0.10e   %0.10e   %0.10e' %
                  (k, H[k], error_norm_1[k], error_norm_inf[k])
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
