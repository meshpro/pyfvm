# -*- coding: utf-8 -*-
#
import numpy
from . import fvm_matrix


class FvmProblem(object):
    def __init__(
            self,
            mesh,
            edge_kernels, vertex_kernels, boundary_kernels, dirichlets,
            edge_matrix_kernels, vertex_matrix_kernels, boundary_matrix_kernels
            ):
        self.mesh = mesh
        self.edge_kernels = edge_kernels
        self.vertex_kernels = vertex_kernels
        self.boundary_kernels = boundary_kernels
        self.dirichlets = dirichlets

        if edge_matrix_kernels or \
           vertex_matrix_kernels or \
           boundary_matrix_kernels:
            self.matrix = fvm_matrix.get_fvm_matrix(
                mesh,
                edge_matrix_kernels,
                vertex_matrix_kernels,
                boundary_matrix_kernels,
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
                edges = self.mesh.get_edges(subdomain)
                edge_nodes = self.mesh.edges['nodes'][edges].T
                numpy.add.at(
                        out,
                        edge_nodes,
                        edge_kernel.eval(u, self.mesh, edges)
                        )

        for vertex_kernel in self.vertex_kernels:
            for subdomain in vertex_kernel.subdomains:
                verts = self.mesh.get_vertices(subdomain)
                out[verts] += vertex_kernel.eval(u, self.mesh, verts)

        for boundary_kernel in self.boundary_kernels:
            for subdomain in boundary_kernel.subdomains:
                verts = self.mesh.get_vertices(subdomain)
                out[verts] += boundary_kernel.eval(u, self.mesh, verts)

        for dirichlet in self.dirichlets:
            verts = self.mesh.get_vertices(dirichlet.subdomain)
            out[verts] = dirichlet.eval(u[verts], self.mesh, verts)

        return out
