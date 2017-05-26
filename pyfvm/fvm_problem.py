# -*- coding: utf-8 -*-
#
import numpy
from . import fvm_matrix


class FvmProblem(object):
    def __init__(
            self,
            mesh,
            edge_kernels, vertex_kernels, face_kernels, dirichlets,
            edge_matrix_kernels, vertex_matrix_kernels, face_matrix_kernels
            ):
        self.mesh = mesh
        self.edge_kernels = edge_kernels
        self.vertex_kernels = vertex_kernels
        self.face_kernels = face_kernels
        self.dirichlets = dirichlets

        if edge_matrix_kernels or \
           vertex_matrix_kernels or \
           face_matrix_kernels:
            self.matrix = fvm_matrix.get_fvm_matrix(
                mesh,
                edge_matrix_kernels,
                vertex_matrix_kernels,
                face_matrix_kernels,
                []  # dirichlets
                )
        else:
            self.matrix = None
        return

    def eval(self, u):

        if self.matrix is None:
            out = numpy.zeros_like(u)
        else:
            out = self.matrix.dot(u)

        for edge_kernel in self.edge_kernels:
            for subdomain in edge_kernel.subdomains:
                cell_mask = self.mesh.get_cell_mask(subdomain)
                numpy.add.at(
                        out,
                        self.mesh.idx_hierarchy,
                        edge_kernel.eval(u, self.mesh, cell_mask)
                        )

        for vertex_kernel in self.vertex_kernels:
            for subdomain in vertex_kernel.subdomains:
                vertex_mask = self.mesh.get_vertex_mask(subdomain)
                out[vertex_mask] += \
                    vertex_kernel.eval(u, self.mesh, vertex_mask)

        for face_kernel in self.face_kernels:
            for subdomain in face_kernel.subdomains:
                face_mask = self.mesh.get_face_mask(subdomain)
                numpy.add(
                    out,
                    face_mask,
                    face_kernel.eval(u, self.mesh, face_mask)
                    )

        for dirichlet in self.dirichlets:
            vertex_mask = self.mesh.get_vertex_mask(dirichlet.subdomain)
            out[vertex_mask] = \
                dirichlet.eval(u[vertex_mask], self.mesh, vertex_mask)

        return out
