# -*- coding: utf-8 -*-
#
import numpy
from scipy import sparse


def tocsr(I, J, E, N):
    import numpy as np
    n = len(I)
    K = np.empty((n,), dtype=np.int64)
    K.view(np.int32).reshape(n, 2).T[...] = J, I
    S = np.argsort(K)
    KS = K[S]
    steps = np.flatnonzero(np.r_[1, np.diff(KS)])
    ED = np.add.reduceat(E[S], steps)
    JD, ID = KS[steps].view(np.int32).reshape(-1, 2).T
    ID = np.searchsorted(ID, np.arange(N+1))
    return sparse.csr_matrix((ED, JD, ID), (N, N))


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
        # matrix = sparse.coo_matrix((V, (I, J)), shape=(n, n))
        # Transform to CSR format for efficiency
        # matrix = matrix.tocsr()
        matrix = tocsr(I, J, V, n)

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
    rhs_V = []
    rhs_I = []

    # Treating the diagonal explicitly makes tocsr() faster at the cost of a
    # bunch of numpy.add.at().
    n = len(mesh.node_coords)
    diag = numpy.zeros(n)

    for edge_kernel in edge_kernels:
        for subdomain in edge_kernel.subdomains:
            if subdomain == 'everywhere':
                cell_ids = mesh.get_cells()
            else:
                cell_ids = mesh.get_cells(subdomain)

            v_mtx, v_rhs, nec = edge_kernel.eval(mesh, cell_ids)

            # assert symmetry in node-edge-cells
            assert (nec[0, 0] == nec[1, 2]).all()
            assert (nec[0, 1] == nec[1, 0]).all()
            assert (nec[0, 2] == nec[1, 1]).all()

            print(nec[0, 0].shape)
            # diagonal entries
            numpy.add.at(diag, nec[0, 0], v_mtx[0, 0, 0] + v_mtx[1, 1, 2])
            numpy.add.at(diag, nec[0, 1], v_mtx[0, 0, 1] + v_mtx[1, 1, 0])
            numpy.add.at(diag, nec[0, 2], v_mtx[0, 0, 2] + v_mtx[1, 1, 1])

            # offdiagonal entries
            V.append(v_mtx[0, 1])
            I.append(nec[0])
            J.append(nec[1])
            #
            V.append(v_mtx[1, 0])
            I.append(nec[1])
            J.append(nec[0])

            # V.append(v_matrix[0, 0])
            # V.append(v_matrix[0, 1])
            # V.append(v_matrix[1, 0])
            # V.append(v_matrix[1, 1])
            #
            # I.append(nec[0])
            # I.append(nec[0])
            # I.append(nec[1])
            # I.append(nec[1])
            #
            # J.append(nec[0])
            # J.append(nec[1])
            # J.append(nec[0])
            # J.append(nec[1])
            #
            # rhs_V.append(v_rhs[0])
            # rhs_V.append(v_rhs[1])
            # rhs_I.append(node_edge_cells[0])
            # rhs_I.append(node_edge_cells[1])

            rhs_V.append(v_rhs[0, 0] + v_rhs[1, 2])
            rhs_V.append(v_rhs[0, 1] + v_rhs[1, 0])
            rhs_V.append(v_rhs[0, 2] + v_rhs[1, 1])
            rhs_I.append(nec[0, 0])
            rhs_I.append(nec[0, 1])
            rhs_I.append(nec[0, 2])

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
                verts = numpy.array(mesh.get_vertices())
            else:
                verts = mesh.get_vertices(subdomain)

            vals_matrix, vals_rhs = vertex_kernel.eval(verts)

            V.append(vals_matrix)
            I.append(verts)
            J.append(verts)

            rhs_V.append(vals_rhs)
            rhs_I.append(verts)

    for face_kernel in face_kernels:
        for subdomain in face_kernel.subdomains:
            verts = mesh.get_vertices(subdomain)
            vals_matrix, vals_rhs = face_kernel.eval(verts)

            V.append(vals_matrix)
            I.append(verts)
            J.append(verts)

            rhs_V.append(vals_rhs)
            rhs_I.append(verts)

    # add diagonal
    I.append(numpy.arange(n))
    J.append(numpy.arange(n))
    V.append(diag)

    # Finally, make V, I, J into 1D-arrays.
    V = numpy.concatenate([v.flat for v in V])
    I = numpy.concatenate([i.flat for i in I])
    J = numpy.concatenate([j.flat for j in J])

    # Assemble rhs
    rhs = numpy.zeros(len(mesh.node_coords))
    for i, v in zip(rhs_I, rhs_V):
        numpy.subtract.at(rhs, i, v)

    return V, I, J, rhs
