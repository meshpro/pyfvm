# -*- coding: utf-8 -*-
#
import numpy
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
                dirichlets,
                compute_rhs=True
                )

        # One unknown per vertex
        n = len(mesh.node_coords)
        self.matrix = sparse.coo_matrix((V, (I, J)), shape=(n, n))
        # Transform to CSR format for efficiency
        self.matrix = self.matrix.tocsr()
        return


def find(lst, a):
    # From <http://stackoverflow.com/a/16685428/353337>
    return [i for i, x in enumerate(lst) if x in a]


def _get_VIJ(
        mesh,
        edge_cores, vertex_cores, boundary_cores, dirichlets,
        compute_rhs=False
        ):
    V = []
    I = []
    J = []
    if compute_rhs:
        rhs = numpy.zeros(len(mesh.node_coords))
    else:
        rhs = None
    for edge_core in edge_cores:
        for subdomain in edge_core.subdomains:
            for k, edge_id in enumerate(mesh.get_edges(subdomain)):
                v0, v1 = mesh.edges['nodes'][edge_id]
                vals_matrix, vals_rhs = edge_core.eval(k)
                V += [
                    vals_matrix[0][0], vals_matrix[0][1],
                    vals_matrix[1][0], vals_matrix[1][1]
                    ]
                I += [v0, v0, v1, v1]
                J += [v0, v1, v0, v1]

                if compute_rhs:
                    rhs[v0] -= vals_rhs[0]
                    rhs[v1] -= vals_rhs[1]

            # # TODO fix those
            # for k in mesh.get_half_edges(subdomain):
            #     k0, k1 = mesh.get_vertices(k)
            #     val = edge_core.eval(k)
            #     V += [vals]
            #     I += [k0, k1]
            #     J += [k0, k1]

    for vertex_core in vertex_cores:
        for subdomain in vertex_core.subdomains:
            for k in mesh.get_vertices(subdomain):
                val_matrix, val_rhs = vertex_core.eval(k)
                V += [val_matrix]
                I += [k]
                J += [k]

                if compute_rhs:
                    rhs[k] -= val_rhs

    for boundary_core in boundary_cores:
        for subdomain in boundary_core.subdomains:
            for k in mesh.get_vertices(subdomain):
                val_matrix, val_rhs = boundary_core.eval(k)
                V += [val_matrix]
                I += [k]
                J += [k]

                if compute_rhs:
                    rhs[k] -= val_rhs

    for dirichlet in dirichlets:
        for subdomain in dirichlet.subdomains:
            boundary_verts = mesh.get_vertices(subdomain)

            # wipe out row k
            indices = find(I, boundary_verts)
            I = numpy.delete(I, indices).tolist()
            J = numpy.delete(J, indices).tolist()
            V = numpy.delete(V, indices).tolist()

            # Add entry 1.0 to the diagonal for each vert
            I.extend(boundary_verts)
            J.extend(boundary_verts)
            V.extend(numpy.ones(len(boundary_verts)))

            if compute_rhs:
                # overwrite rhs
                for k in mesh.get_vertices(subdomain):
                    rhs[k] = dirichlet.eval(k)

    return V, I, J, rhs
