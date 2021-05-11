import matplotlib.pyplot as plt
import meshplex
import numpy as np
import sympy


def perform_convergence_tests(discrete_solver, exact_sol, get_mesh, rng, verbose=False):
    n = len(rng)
    H = np.empty(n)
    error_norm_1 = np.empty(n)
    order_1 = np.empty(n - 1)
    error_norm_inf = np.empty(n)
    order_inf = np.empty(n - 1)

    if verbose:
        print(79 * "-")
        print(
            "k"
            + 5 * " "
            + "num verts"
            + 4 * " "
            + "max edge length"
            + 4 * " "
            + "||error||_1"
            + 8 * " "
            + "||error||_inf"
        )
        print(38 * " " + "(order)" + 12 * " " + "(order)")
        print(79 * "-")

    # Add "zero" to all entities. This later gets translated into
    # np.zeros with the appropriate length, making sure that scalar
    # terms in the lambda expression correctly return np.arrays.
    zero = sympy.Symbol("zero")
    x = sympy.DeferredVector("x")
    # See <http://docs.sympy.org/dev/modules/utilities/lambdify.html>.
    array2array = [{"ImmutableMatrix": np.array}, "numpy"]
    exact_eval = sympy.lambdify((x, zero), exact_sol(x), modules=array2array)

    for k in rng:
        mesh = get_mesh(k)
        # get max edge length
        H[k] = np.sqrt(mesh.ei_dot_ei.max())

        u = discrete_solver(mesh)

        zero = np.zeros(len(mesh.points))
        error = u - exact_eval(mesh.points.T, zero)

        # import meshio
        # meshio.write(
        #     'sol%d.vtu' % k,
        #     mesh.points, {'triangle': mesh.cells["points']},
        #     point_data={'x': x, 'error': error},
        #     )

        error_norm_1[k] = np.sum(abs(mesh.control_volumes * error))
        error_norm_inf[k] = max(abs(error))

        # numerical orders of convergence
        if k > 0:
            order_1[k - 1] = np.log(error_norm_1[k - 1] / error_norm_1[k]) / np.log(
                H[k - 1] / H[k]
            )
            order_inf[k - 1] = np.log(
                error_norm_inf[k - 1] / error_norm_inf[k]
            ) / np.log(H[k - 1] / H[k])
            if verbose:
                print
                print(
                    (38 * " " + "%0.5f" + 12 * " " + "%0.5f")
                    % (order_1[k - 1], order_inf[k - 1])
                )
                print

        if verbose:
            num_nodes = len(mesh.points)
            print(
                "%2d    %5.3e    %0.10e   %0.10e   %0.10e"
                % (k, num_nodes, H[k], error_norm_1[k], error_norm_inf[k])
            )

    return H, error_norm_1, error_norm_inf, order_1, order_inf


def show_error_data(*args, **kwargs):
    plot_error_data(*args, **kwargs)
    plt.show()


def plot_error_data(H, error_norm_1, error_norm_inf):
    # plot error data
    plt.loglog(H, error_norm_1, "xk", label="||error||_1")
    plt.loglog(H, error_norm_inf, "ok", label="||error||_inf")

    # plot 2nd order indicator
    e0 = max(error_norm_1[0], error_norm_inf[0])
    order = 2
    plt.loglog(
        [H[0], H[-1]],
        [10 * e0, 10 * e0 * (H[-1] / H[0]) ** order],
        "--k",
        label="2nd order",
    )

    plt.legend(loc="upper left")


# def get_ball_mesh(k):
#     import dolfin
#     import mshr
#     h = 0.5**(k+2)
#     c = mshr.Sphere(dolfin.Point(0., 0., 0.), 1.0, int(2*pi / h))
#     m = mshr.generate_mesh(c, 2.0 / h)
#     return meshplex.MeshTetra(
#             m.coordinates(),
#             m.cells(),
#             )


def get_ball_mesh(k):
    import pygmsh

    h = 0.5 ** (k + 1)
    geom = pygmsh.built_in.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, lcar=h)
    mesh = pygmsh.generate_mesh(geom, verbose=False)
    cells = mesh.get_cells_type("tetra")
    # toss away unused points
    uvertices, uidx = np.unique(cells, return_inverse=True)
    cells = uidx.reshape(cells.shape)
    points = mesh.points[uvertices]
    return meshplex.MeshTetra(points, cells)


# def get_disk_mesh(k):
#     import dolfin
#     import mshr
#     from numpy import pi
#     h = 0.5**k
#     # cell_size = 2 * pi / num_Boundary()_points
#     c = mshr.Circle(dolfin.Point(0., 0., 0.), 1, int(2*pi / h))
#     # cell_size = 2 * bounding_box_radius / res
#     m = mshr.generate_mesh(c, 2.0 / h)
#     coords = m.coordinates()
#     coords = np.c_[coords, np.zeros(len(coords))]
#     return meshplex.MeshTri(coords, m.cells())


def get_disk_mesh(k):
    import meshzoo

    points, cells = meshzoo.disk(6, k + 1)
    out = meshplex.MeshTri(points, cells)
    # out.show()
    return out
