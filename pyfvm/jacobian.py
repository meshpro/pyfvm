# -*- coding: utf-8 -*-
#
import numpy
from scipy import sparse


class Jacobian(object):
    def __init__(
            self,
            mesh,
            edge_kernels, vertex_kernels, face_kernels, dirichlets
            ):
        self.mesh = mesh
        self.edge_kernels = edge_kernels
        self.vertex_kernels = vertex_kernels
        self.face_kernels = face_kernels
        self.dirichlets = dirichlets
        return

    def get_linear_operator(self, u):
        V, I, J = _get_VIJ(
                self.mesh,
                u,
                self.edge_kernels, self.vertex_kernels, self.face_kernels
                )

        # One unknown per vertex
        n = len(self.mesh.node_coords)
        matrix = sparse.coo_matrix((V, (I, J)), shape=(n, n))
        # Transform to CSR format for efficiency
        matrix = matrix.tocsr()

        # Apply Dirichlet conditions.
        d = matrix.diagonal()
        for dirichlet in self.dirichlets:
            verts = self.mesh.get_vertices(dirichlet.subdomain)
            # Set all Dirichlet rows to 0.
            for i in verts:
                matrix.data[matrix.indptr[i]:matrix.indptr[i+1]] = 0.0

            # Set the diagonal.
            d[verts] = dirichlet.eval(u[verts], self.mesh, verts)

        matrix.setdiag(d)

        return matrix


def _get_VIJ(
        mesh,
        u,
        edge_kernels, vertex_kernels, face_kernels
        ):
    V = []
    I = []
    J = []

    for edge_kernel in edge_kernels:
        for subdomain in edge_kernel.subdomains:
            cell_ids = mesh.get_cells(subdomain)
            v_matrix = edge_kernel.eval(u, mesh, cell_ids)

            V.append(v_matrix[0, 0].flatten())
            V.append(v_matrix[0, 1].flatten())
            V.append(v_matrix[1, 0].flatten())
            V.append(v_matrix[1, 1].flatten())

            I.append(mesh.idx_hierarchy[0].flatten())
            I.append(mesh.idx_hierarchy[0].flatten())
            I.append(mesh.idx_hierarchy[1].flatten())
            I.append(mesh.idx_hierarchy[1].flatten())

            J.append(mesh.idx_hierarchy[0].flatten())
            J.append(mesh.idx_hierarchy[1].flatten())
            J.append(mesh.idx_hierarchy[0].flatten())
            J.append(mesh.idx_hierarchy[1].flatten())

    for vertex_kernel in vertex_kernels:
        for subdomain in vertex_kernel.subdomains:
            verts = mesh.get_vertices(subdomain)
            vals_matrix = vertex_kernel.eval(u, mesh, verts)

            if verts == slice(None, None, None):
                verts = numpy.arange(len(vals_matrix))
            V.append(vals_matrix)
            I.append(verts)
            J.append(verts)

    for face_kernel in face_kernels:
        for subdomain in face_kernel.subdomains:
            faces = mesh.get_vertices(subdomain)
            vals_matrix = face_kernel.eval(u, mesh, faces)

            V.append(vals_matrix)
            I.append(faces)
            J.append(faces)

    # Finally, make V, I, J into 1D-arrays.
    V = numpy.concatenate(V)
    I = numpy.concatenate(I)
    J = numpy.concatenate(J)

    return V, I, J
