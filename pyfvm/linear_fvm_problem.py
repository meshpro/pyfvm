# -*- coding: utf-8 -*-
#
import numpy
from scipy import sparse


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

        # Apply Dirichlet conditions.
        d = self.matrix.diagonal()
        for dirichlet in dirichlets:
            verts = mesh.get_vertices(dirichlet.subdomain)
            # Set all Dirichlet rows to 0.
            for i in verts:
                self.matrix.data[self.matrix.indptr[i]:self.matrix.indptr[i+1]] = 0.0

            # Set the diagonal and RHS.
            coeff, rhs = dirichlet.eval(verts)
            d[verts] = coeff
            self.rhs[verts] = rhs

        self.matrix.setdiag(d)

        return


def _get_VIJ(
        mesh,
        edge_kernels, vertex_kernels, boundary_kernels,
        compute_rhs=False
        ):
    V = []
    I = []
    J = []
    rhs_V = []
    rhs_I = []

    for edge_kernel in edge_kernels:
        for subdomain in edge_kernel.subdomains:
            edges = mesh.get_edges(subdomain)
            edge_nodes = mesh.edges['nodes'][edges]

            v_matrix, v_rhs = edge_kernel.eval(edges)

            # if dot() is used in the expression, the shape of of v_matrix will
            # be (2, 2, 1, k) instead of (2, 2, k).
            if len(v_matrix.shape) == 4:
                assert v_matrix.shape[2] == 1
                V.append(v_matrix[0, 0, 0, :])
                V.append(v_matrix[0, 1, 0, :])
                V.append(v_matrix[1, 0, 0, :])
                V.append(v_matrix[1, 1, 0, :])
            else:
                V.append(v_matrix[0, 0, :])
                V.append(v_matrix[0, 1, :])
                V.append(v_matrix[1, 0, :])
                V.append(v_matrix[1, 1, :])

            I.append(edge_nodes[:, 0])
            I.append(edge_nodes[:, 0])
            I.append(edge_nodes[:, 1])
            I.append(edge_nodes[:, 1])

            J.append(edge_nodes[:, 0])
            J.append(edge_nodes[:, 1])
            J.append(edge_nodes[:, 0])
            J.append(edge_nodes[:, 1])

            if compute_rhs:
                rhs_V.append(v_rhs[0])
                rhs_V.append(v_rhs[1])
                rhs_I.append(edge_nodes[:, 0])
                rhs_I.append(edge_nodes[:, 1])

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

            V.append(vals_matrix)
            I.append(verts)
            J.append(verts)

            if compute_rhs:
                rhs_V.append(vals_rhs)
                rhs_I.append(verts)

    for boundary_kernel in boundary_kernels:
        for subdomain in boundary_kernel.subdomains:
            verts = mesh.get_vertices(subdomain)
            vals_matrix, vals_rhs = boundary_kernel.eval(verts)

            V.append(vals_matrix)
            I.append(verts)
            J.append(verts)

            if compute_rhs:
                rhs_V.append(vals_rhs)
                rhs_I.append(verts)

    # Finally, make V, I, J into 1D-arrays.
    V = numpy.concatenate(V)
    I = numpy.concatenate(I)
    J = numpy.concatenate(J)

    # Assemble rhs
    if compute_rhs:
        rhs = numpy.zeros(len(mesh.node_coords))
        numpy.subtract.at(
                rhs,
                numpy.concatenate(rhs_I),
                numpy.concatenate(rhs_V)
                )
    else:
        rhs = None

    return V, I, J, rhs
