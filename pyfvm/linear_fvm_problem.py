# -*- coding: utf-8 -*-
#
import numpy
from scipy import sparse


def get_linear_fvm_problem(
        mesh,
        edge_kernels, vertex_kernels, boundary_kernels, dirichlets
        ):
        V, I, J, rhs = _get_VIJ(
                mesh,
                edge_kernels,
                vertex_kernels,
                boundary_kernels
                )

        # One unknown per vertex
        n = len(mesh.node_coords)
        matrix = sparse.coo_matrix((V, (I, J)), shape=(n, n))
        # Transform to CSR format for efficiency
        matrix = matrix.tocsr()

        # Apply Dirichlet conditions.
        d = matrix.diagonal()
        for dirichlet in dirichlets:
            print(dirichlet.subdomain)
            print(type(dirichlet.subdomain))
            verts = mesh.get_vertices(dirichlet.subdomain)
            # Set all Dirichlet rows to 0.
            for i in verts:
                matrix.data[matrix.indptr[i]:matrix.indptr[i+1]] = 0.0

            # Set the diagonal and RHS.
            coeff, rhs_vals = dirichlet.eval(verts)
            d[verts] = coeff
            rhs[verts] = rhs_vals

        matrix.setdiag(d)

        return matrix, rhs


def _get_VIJ(
        mesh,
        edge_kernels, vertex_kernels, boundary_kernels
        ):
    V = []
    I = []
    J = []
    rhs_V = []
    rhs_I = []

    for edge_kernel in edge_kernels:
        for subdomain in edge_kernel.subdomains:
            if subdomain == 'everywhere':
                cell_ids = mesh.get_cells()
            else:
                cell_ids = mesh.get_cells(subdomain)

            v_matrix, v_rhs, cell_edge_nodes = edge_kernel.eval(mesh, cell_ids)

            # if dot() is used in the expression, the shape of of v_matrix will
            # be (2, 2, 1, k) instead of (2, 2, 871, k).
            if len(v_matrix.shape) == 5:
                assert v_matrix.shape[2] == 1
                V.append(v_matrix[0, 0, 0])
                V.append(v_matrix[0, 1, 0])
                V.append(v_matrix[1, 0, 0])
                V.append(v_matrix[1, 1, 0])
            else:
                V.append(v_matrix[0, 0])
                V.append(v_matrix[0, 1])
                V.append(v_matrix[1, 0])
                V.append(v_matrix[1, 1])

            I.append(cell_edge_nodes[0])
            I.append(cell_edge_nodes[0])
            I.append(cell_edge_nodes[1])
            I.append(cell_edge_nodes[1])

            J.append(cell_edge_nodes[0])
            J.append(cell_edge_nodes[1])
            J.append(cell_edge_nodes[0])
            J.append(cell_edge_nodes[1])

            rhs_V.append(v_rhs[0])
            rhs_V.append(v_rhs[1])
            rhs_I.append(cell_edge_nodes[0])
            rhs_I.append(cell_edge_nodes[1])

            # # TODO fix those
            # for k in mesh.get_half_edges(subdomain):
            #     k0, k1 = mesh.get_vertices(k)
            #     val = edge_kernel.eval(k)
            #     V += [vals]
            #     I += [k0, k1]
            #     J += [k0, k1]

    for vertex_kernel in vertex_kernels:
        for subdomain in vertex_kernel.subdomains:
            if subdomain == 'everywhere':
                verts = numpy.array(mesh.get_vertices())
            else:
                verts = mesh.get_vertices(subdomain)

            vals_matrix, vals_rhs = vertex_kernel.eval(verts)

            V.append(vals_matrix)
            I.append(verts)
            J.append(verts)

            rhs_V.append(vals_rhs)
            rhs_I.append(verts)

    for boundary_kernel in boundary_kernels:
        for subdomain in boundary_kernel.subdomains:
            verts = mesh.get_vertices(subdomain)
            vals_matrix, vals_rhs = boundary_kernel.eval(verts)

            V.append(vals_matrix)
            I.append(verts)
            J.append(verts)

            rhs_V.append(vals_rhs)
            rhs_I.append(verts)

    # Finally, make V, I, J into 1D-arrays.
    V = numpy.concatenate([v.flat for v in V])
    I = numpy.concatenate([i.flat for i in I])
    J = numpy.concatenate([j.flat for j in J])

    # Assemble rhs
    rhs = numpy.zeros(len(mesh.node_coords))
    for i, v in zip(rhs_I, rhs_V):
        numpy.subtract.at(rhs, i, v)

    return V, I, J, rhs
