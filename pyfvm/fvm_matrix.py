import numpy
from scipy import sparse


def get_fvm_matrix(
    mesh, edge_kernels=None, vertex_kernels=None, face_kernels=None, dirichlets=None
):
    edge_kernels = [] if edge_kernels is None else edge_kernels
    vertex_kernels = [] if vertex_kernels is None else vertex_kernels
    face_kernels = [] if face_kernels is None else face_kernels
    dirichlets = [] if dirichlets is None else dirichlets

    V, I, J = _get_VIJ(mesh, edge_kernels, vertex_kernels, face_kernels)

    # One unknown per vertex
    n = len(mesh.points)
    matrix = sparse.coo_matrix((V, (I, J)), shape=(n, n))
    # Transform to CSR format for efficiency
    matrix = matrix.tocsr()

    # Apply Dirichlet conditions.
    d = matrix.diagonal()
    for dirichlet in dirichlets:
        verts = mesh.get_vertices(dirichlet.subdomain)
        # Set all Dirichlet rows to 0.
        for i in verts:
            matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]] = 0.0

        # Set the diagonal and RHS.
        d[verts] = dirichlet.eval(mesh, verts)

    matrix.setdiag(d)

    return matrix


def _get_VIJ(mesh, edge_kernels, vertex_kernels, face_kernels):
    V = []
    I_ = []
    J = []

    for edge_kernel in edge_kernels:
        for subdomain in edge_kernel.subdomains:
            cell_mask = mesh.get_cell_mask(subdomain)

            v_matrix = edge_kernel.eval(mesh, cell_mask)

            V.append(v_matrix[0, 0].flatten())
            V.append(v_matrix[0, 1].flatten())
            V.append(v_matrix[1, 0].flatten())
            V.append(v_matrix[1, 1].flatten())

            I_.append(mesh.idx_hierarchy[0].flatten())
            I_.append(mesh.idx_hierarchy[0].flatten())
            I_.append(mesh.idx_hierarchy[1].flatten())
            I_.append(mesh.idx_hierarchy[1].flatten())

            J.append(mesh.idx_hierarchy[0].flatten())
            J.append(mesh.idx_hierarchy[1].flatten())
            J.append(mesh.idx_hierarchy[0].flatten())
            J.append(mesh.idx_hierarchy[1].flatten())

    # TODO
    # for vertex_kernel in vertex_kernels:
    #     for subdomain in vertex_kernel.subdomains:
    #         vertex_mask = mesh.get_vertex_mask(subdomain)
    #         vals_matrix = vertex_kernel.eval(mesh, vertex_mask)

    #         V.append(vals_matrix)
    #         I_.append(verts)
    #         J.append(verts)

    for face_kernel in face_kernels:
        for subdomain in face_kernel.subdomains:
            face_mask = mesh.get_face_mask(subdomain)
            vals_matrix = face_kernel.eval(mesh, face_mask)

            ids = mesh.idx_hierarchy[..., face_mask]
            V.append(vals_matrix)
            I_.append(ids)
            J.append(ids)

    # Finally, make V, I, J into 1D-arrays.
    V = numpy.concatenate(V)
    I_ = numpy.concatenate(I_)
    J = numpy.concatenate(J)

    return V, I_, J
