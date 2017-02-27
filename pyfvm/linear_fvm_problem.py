# -*- coding: utf-8 -*-
#
import numpy
from scipy import sparse


def get_linear_fvm_problem(
        mesh,
        edge_kernels, vertex_kernels, face_kernels, dirichlets
        ):
        V, I, J, rhs = _get_VIJ(
                mesh,
                edge_kernels,
                vertex_kernels,
                face_kernels
                )

        # One unknown per vertex
        n = len(mesh.node_coords)
        # Transform to CSR format for efficiency
        matrix = sparse.coo_matrix((V, (I, J)), shape=(n, n))
        matrix = matrix.tocsr()

        # Apply Dirichlet conditions.
        d = matrix.diagonal()
        for dirichlet in dirichlets:
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
        edge_kernels, vertex_kernels, face_kernels
        ):
    V = []
    I = []
    J = []
    n = len(mesh.node_coords)
    # Treating the diagonal explicitly makes tocsr() faster at the cost of a
    # bunch of numpy.add.at().
    diag = numpy.zeros(n)
    #
    rhs = numpy.zeros(n)

    for edge_kernel in edge_kernels:
        for subdomain in edge_kernel.subdomains:
            if subdomain == 'everywhere':
                cell_ids = mesh.get_cells()
            else:
                cell_ids = mesh.get_cells(subdomain)

            v_mtx, v_rhs, nec = edge_kernel.eval(mesh, cell_ids)

            # Diagonal entries.
            # Manually sum up the entries corresponding to the same i, j first.
            for c, i in zip(mesh.cells['nodes'].T, mesh.local_idx_inv):
                numpy.add.at(
                    diag,
                    c,
                    sum([v_mtx[t[0]][t[0]][t[1:]] for t in i])
                    )

            # offdiagonal entries
            V.append(v_mtx[0][1])
            I.append(nec[0])
            J.append(nec[1])
            #
            V.append(v_mtx[1][0])
            I.append(nec[1])
            J.append(nec[0])

            # Right-hand side.
            try:
                for c, i in zip(mesh.cells['nodes'].T, mesh.local_idx_inv):
                    numpy.subtract.at(
                        rhs,
                        c,
                        sum([v_rhs[t[0]][t[1:]] for t in i])
                        )
            except TypeError:
                # v_rhs probably integers/floats
                if v_rhs[0] != 0:
                    # FIXME these at operations seem really slow with v_rhs
                    #       not being of type numpy.ndarray
                    numpy.subtract.at(rhs, nec[0], v_rhs[0])
                if v_rhs[1] != 0:
                    numpy.subtract.at(rhs, nec[1], v_rhs[1])

            # if dot() is used in the expression, the shape of of v_matrix will
            # be (2, 2, 1, k) instead of (2, 2, 871, k).
            # if len(v_matrix.shape) == 5:
            #     assert v_matrix.shape[2] == 1
            #     V.append(v_matrix[0, 0, 0])
            #     V.append(v_matrix[0, 1, 0])
            #     V.append(v_matrix[1, 0, 0])
            #     V.append(v_matrix[1, 1, 0])
            # else:

    for vertex_kernel in vertex_kernels:
        for subdomain in vertex_kernel.subdomains:
            if subdomain == 'everywhere':
                verts = mesh.get_vertices()
            else:
                verts = mesh.get_vertices(subdomain)

            vals_matrix, vals_rhs = vertex_kernel.eval(verts)

            if verts == numpy.s_[:]:
                diag += vals_matrix
                rhs -= vals_rhs
            else:
                # numpy.add.at(diag, verts, vals_matrix)
                # numpy.subtract.at(rhs, verts, vals_rhs)
                diag[verts] += vals_matrix
                rhs[verts] -= vals_rhs

    for face_kernel in face_kernels:
        for subdomain in face_kernel.subdomains:
            verts = mesh.get_vertices(subdomain)
            vals_matrix, vals_rhs = face_kernel.eval(verts)

            V.append(vals_matrix)
            I.append(verts)
            J.append(verts)

            if verts == numpy.s_[:]:
                rhs -= vals_rhs
            else:
                numpy.subtract.at(rhs, verts, vals_rhs)

    # add diagonal
    I.append(numpy.arange(n))
    J.append(numpy.arange(n))
    V.append(diag)

    # Finally, make V, I, J into 1D-arrays.
    V = numpy.concatenate([v.flat for v in V])
    I = numpy.concatenate([i.flat for i in I])
    J = numpy.concatenate([j.flat for j in J])

    return V, I, J, rhs
