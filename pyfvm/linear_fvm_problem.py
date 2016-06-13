from scipy import sparse


class LinearFvmProblem(object):
    def __init__(
            self,
            mesh,
            edge_cores, vertex_cores, boundary_cores, dirichlets
            ):
        self.mesh = mesh
        self.edge_cores = edge_cores
        self.vertex_cores = vertex_cores
        self.boundary_cores = boundary_cores
        self.dirichlets = dirichlets

        V, I, J, self.rhs = _get_VIJ(
                mesh,
                edge_cores,
                vertex_cores,
                boundary_cores,
                dirichlets
                )
        self.matrix = sparse.coo_matrix((V, (I, J)), shape=(n, n))
        return


def _get_VIJ(
        mesh,
        edge_cores, vertex_cores, boundary_cores, dirichlets,
        compute_rhs=False
        ):
    V = []
    I = []
    J = []
    if compute_rhs:
        rhs = numpy.zeros(mesh.n)
    else:
        rhs = None
    for edge_core in edge_cores:
        for subdomain in edge_core.subdomains:
            for k in mesh.get_edges(subdomain):
                # TODO fix this
                k0, k1 = mesh.get_edge_vertices(k)
                vals_matrix, vals_rhs = edge_core.eval(k)
                V += [vals]
                I += [k0, k1]
                J += [k0, k1]

                if compute_rhs:
                    rhs[k0] += vals_rhs[0]
                    rhs[k1] += vals_rhs[1]

            # # TODO fix those
            # for k in mesh.get_half_edges(subdomain):
            #     k0, k1 = mesh.get_vertices(k)
            #     val = edge_core.eval(k)
            #     V += [vals]
            #     I += [k0, k1]
            #     J += [k0, k1]

    for vertex_core in vertex_cores:
        raise NotImplemented('vertex core')

    for dirichlet in dirichlets:
        for subdomain in dirichlet.subdomains:
            for k in mesh.get_vertices(subdomain):
                # TODO wipe out row k

                rhs = vals_rhs[0]

                # overwrite rhs
                if compute_rhs:
                    rhs[k] = dirichlet.eval(k)

    return V, I, J, rhs
