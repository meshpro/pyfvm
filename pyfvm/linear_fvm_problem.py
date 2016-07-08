# -*- coding: utf-8 -*-
#
import numpy
from scipy import sparse


def _wipe_row_csr(matrix, i):
    '''Wipes a row of a matrix in CSR format and puts 1.0 on the diagonal.
    '''
    assert isinstance(matrix, sparse.csr_matrix)

    n = matrix.indptr[i+1] - matrix.indptr[i]
    assert n > 0

    matrix.data[matrix.indptr[i]+1:-n+1] = matrix.data[matrix.indptr[i+1]:]
    matrix.data[matrix.indptr[i]] = 1.0
    matrix.data = matrix.data[:-n+1]

    matrix.indices[matrix.indptr[i]+1:-n+1] = \
        matrix.indices[matrix.indptr[i+1]:]
    matrix.indices[matrix.indptr[i]] = i
    matrix.indices = matrix.indices[:-n+1]

    matrix.indptr[i+1:] -= n-1

    return


class LinearFvmProblem(object):
    def __init__(
            self,
            mesh,
            edge_kernels, vertex_kernels, boundary_kernels, dirichlets
            ):
        self.mesh = mesh
        self.edge_kernels = edge_kernels
        self.vertex_kernels = vertex_kernels
        self.boundary_kernels = boundary_kernels
        self.dirichlets = dirichlets

        V, I, J, self.rhs = _get_VIJ(
                mesh,
                edge_kernels,
                vertex_kernels,
                boundary_kernels,
                compute_rhs=True
                )

        # One unknown per vertex
        n = len(mesh.node_coords)
        self.matrix = sparse.coo_matrix((V, (I, J)), shape=(n, n))
        # Transform to CSR format for efficiency
        self.matrix = self.matrix.tocsr()

        for dirichlet in dirichlets:
            for subdomain in dirichlet.subdomains:
                boundary_verts = mesh.get_vertices(subdomain)

                for k in boundary_verts:
                    _wipe_row_csr(self.matrix, k)

                # overwrite rhs
                self.rhs[boundary_verts] = dirichlet.eval(boundary_verts)

        return


def _get_VIJ(
        mesh,
        edge_kernels, vertex_kernels, boundary_kernels,
        compute_rhs=False
        ):
    V = []
    I = []
    J = []
    if compute_rhs:
        rhs = numpy.zeros(len(mesh.node_coords))
    else:
        rhs = None
    for edge_kernel in edge_kernels:
        for subdomain in edge_kernel.subdomains:
            edges = mesh.get_edges(subdomain)
            edge_nodes = mesh.edges['nodes'][edges]
            v_matrix, v_rhs = edge_kernel.eval(edges)

            v = numpy.vstack([
                v_matrix[0, 0, :], v_matrix[0, 1, :],
                v_matrix[1, 0, :], v_matrix[1, 1, :]
                ]).T.flatten()
            V += list(v)

            if compute_rhs:
                numpy.subtract.at(
                        rhs,
                        edge_nodes[edges, 0],
                        v_rhs[0]
                        )
                numpy.subtract.at(
                        rhs,
                        edge_nodes[edges, 1],
                        v_rhs[1]
                        )

            i = numpy.vstack([
                edge_nodes[:, 0], edge_nodes[:, 0],
                edge_nodes[:, 1], edge_nodes[:, 1]
                ]).T.flatten()
            I += list(i)

            j = numpy.vstack([
                edge_nodes[:, 0], edge_nodes[:, 1],
                edge_nodes[:, 0], edge_nodes[:, 1]
                ]).T.flatten()
            J += list(j)

            # # TODO fix those
            # for k in mesh.get_half_edges(subdomain):
            #     k0, k1 = mesh.get_vertices(k)
            #     val = edge_kernel.eval(k)
            #     V += [vals]
            #     I += [k0, k1]
            #     J += [k0, k1]

    for vertex_kernel in vertex_kernels:
        for subdomain in vertex_kernel.subdomains:
            verts = mesh.get_vertices(subdomain)
            vals_matrix, vals_rhs = vertex_kernel.eval(verts)
            V += list(vals_matrix)
            I += list(verts)
            J += list(verts)
            if compute_rhs:
                numpy.subtract.at(
                        rhs,
                        verts,
                        vals_rhs
                        )

    for boundary_kernel in boundary_kernels:
        for subdomain in boundary_kernel.subdomains:
            verts = mesh.get_vertices(subdomain)
            vals_matrix, vals_rhs = boundary_kernel.eval(verts)
            V += list(vals_matrix)
            I += list(verts)
            J += list(verts)
            if compute_rhs:
                numpy.subtract.at(
                        rhs,
                        verts,
                        vals_rhs
                        )

    return V, I, J, rhs
