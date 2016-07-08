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

        V = numpy.array(V)

        # Apply Dirichlet conditions.
        for dirichlet in dirichlets:
            for subdomain in dirichlet.subdomains:
                # First set all Dirichlet rows to 0.
                verts = mesh.get_vertices(subdomain)
                mask = numpy.in1d(I, verts)
                V[mask] = 0.0
                # Now add 1.0 to the diagonal for each Dirichlet.
                I = numpy.append(I, verts)
                J = numpy.append(J, verts)
                V = numpy.append(V, numpy.ones(len(verts)))

                # Set the RHS.
                self.rhs[verts] = dirichlet.eval(verts)

        # One unknown per vertex
        n = len(mesh.node_coords)
        self.matrix = sparse.coo_matrix((V, (I, J)), shape=(n, n))
        # Transform to CSR format for efficiency
        self.matrix = self.matrix.tocsr()

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
            V.append(v)

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
            I.append(i)

            j = numpy.vstack([
                edge_nodes[:, 0], edge_nodes[:, 1],
                edge_nodes[:, 0], edge_nodes[:, 1]
                ]).T.flatten()
            J.append(j)

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
                numpy.subtract.at(
                        rhs,
                        verts,
                        vals_rhs
                        )

    for boundary_kernel in boundary_kernels:
        for subdomain in boundary_kernel.subdomains:
            verts = mesh.get_vertices(subdomain)
            vals_matrix, vals_rhs = boundary_kernel.eval(verts)
            V.append(vals_matrix)
            I.append(verts)
            J.append(verts)
            if compute_rhs:
                numpy.subtract.at(
                        rhs,
                        verts,
                        vals_rhs
                        )

    # Finally, make V, I, J into 1D-arrays.
    V = numpy.concatenate(V)
    I = numpy.concatenate(I)
    J = numpy.concatenate(J)

    return V, I, J, rhs
