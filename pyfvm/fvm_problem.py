# -*- coding: utf-8 -*-
#
import numpy
from scipy import sparse


class FvmProblem(object):
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
        return

    def eval(self, u):

        out = numpy.zeros_like(u)

        for edge_kernel in self.edge_kernels:
            for subdomain in edge_kernel.subdomains:
                edges = self.mesh.get_edges(subdomain)
                edge_nodes = self.mesh.edges['nodes'][edges].T
                numpy.add.at(
                        out,
                        edge_nodes,
                        edge_kernel.eval(u, edges)
                        )

        for vertex_kernel in self.vertex_kernels:
            for subdomain in vertex_kernel.subdomains:
                verts = self.mesh.get_vertices(subdomain)
                out[verts] += vertex_kernel.eval(u, verts)

        for boundary_kernel in self.boundary_kernels:
            for subdomain in boundary_kernel.subdomains:
                verts = self.mesh.get_vertices(subdomain)
                out[verts] += boundary_kernel.eval(u, verts)

        for dirichlet in self.dirichlets:
            verts = self.mesh.get_vertices(dirichlet.subdomain)
            out[verts] = dirichlet.eval(u[verts], verts)

        return out
