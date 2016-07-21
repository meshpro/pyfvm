# -*- coding: utf-8 -*-
#
import meshio
import numpy

__all__ = []


def _row_dot(a, b):
    # http://stackoverflow.com/a/26168677/353337
    return numpy.einsum('ij, ij->i', a, b)


class _base_mesh(object):

    def __init__(self,
                 nodes,
                 cells_nodes
                 ):
        self.node_coords = nodes
        return

    def write(self,
              filename,
              point_data=None,
              cell_data=None,
              field_data=None
              ):
        if self.node_coords.shape[1] == 2:
            n = len(self.node_coords)
            a = numpy.ascontiguousarray(
                numpy.c_[self.node_coords, numpy.zeros(n)]
                )
        else:
            a = self.node_coords

        if self.cells['nodes'].shape[1] == 3:
            cell_type = 'triangle'
        elif self.cells['nodes'].shape[1] == 4:
            cell_type = 'tetra'
        else:
            raise RuntimeError('Only triangles/tetrahedra supported')

        meshio.write(
            filename,
            a,
            {cell_type: self.cells['nodes']},
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )

    def compute_edge_lengths(self):
        edges = self.node_coords[self.edges['nodes'][:, 1]] \
            - self.node_coords[self.edges['nodes'][:, 0]]
        self.edge_lengths = numpy.sqrt(_row_dot(edges, edges))
        return

    def compute_tri_areas_and_ce_ratios(self, e0, e1, e2):
        '''Given triangles (specified by their edges) and vertex
        coordinates, this routine will return the triangle areas and the signed
        distances of the triangle circumcenters to the edge midpoints.
        '''
        # The covolume-edge ratios for the edges of each cell is the solution
        # of the equation system
        #
        # |simplex| ||u||^2 = \sum_i \alpha_i <u,e_i> <e_i,u>,
        #
        # where alpha_i are the covolume contributions for the edges.
        #
        # This equation system to hold for all vectors u in the plane spanned
        # by the edges, particularly by the edges themselves.
        #
        # For triangles, the exact solution of the system is
        #
        #  x_1 = <e_2, e_3> / <e1 x e2, e1 x e3> * |simplex|;
        #
        # see <http://math.stackexchange.com/a/1855380/36678>.

        e0_cross_e1 = numpy.cross(e0, e1)
        e1_cross_e2 = numpy.cross(e1, e2)
        e2_cross_e0 = numpy.cross(e2, e0)
        cell_volumes = 0.5 * numpy.sqrt(
                _row_dot(e0_cross_e1, e0_cross_e1)
                )

        a = _row_dot(e1, e2) / _row_dot(e0_cross_e1, -e2_cross_e0)
        b = _row_dot(e2, e0) / _row_dot(e1_cross_e2, -e0_cross_e1)
        c = _row_dot(e0, e1) / _row_dot(e2_cross_e0, -e1_cross_e2)

        # Note
        # ----
        #
        # There is an alternative formulation in terms of dot-products via
        #
        #   <e1 x e2, e1 x e3> = <e1, e1> <e2, e3> - <e1, e2> <e1, e3>.
        #
        # With this, the solution can be expressed as
        #
        #  x_1 / |simplex| * <e1, e1>
        #    = <e1, e1> <e_2, e_3> / (<e1, e1> <e2, e3> - <e1, e2> <e1, e3>)
        #
        # It doesn't matter much which cross product we take for computing the
        # cell volume.
        # However, this approach is less favorable in terms of round-off
        # errors: For almost degenerate triangles, the difference in the
        # denominator is small, but the two values are large. This will lead to
        # significant round-off in the denominator.
        #
        # e0_dot_e0 = _row_dot(cells_edges[:, 0, :], cells_edges[:, 0, :])
        # e0_dot_e1 = _row_dot(cells_edges[:, 0, :], cells_edges[:, 1, :])
        # e0_dot_e2 = _row_dot(cells_edges[:, 0, :], cells_edges[:, 2, :])
        # e1_dot_e1 = _row_dot(cells_edges[:, 1, :], cells_edges[:, 1, :])
        # e1_dot_e2 = _row_dot(cells_edges[:, 1, :], cells_edges[:, 2, :])
        # e2_dot_e2 = _row_dot(cells_edges[:, 2, :], cells_edges[:, 2, :])
        #
        # a00 = e0_dot_e0 * e1_dot_e2
        # a = e1_dot_e2 / (a00 - (e0_dot_e1 * e0_dot_e2))
        # b00 = e1_dot_e1 * e0_dot_e2
        # b = e0_dot_e2 / (b00 - (e0_dot_e1 * e1_dot_e2))
        # c00 = e2_dot_e2 * e0_dot_e1
        # c = e0_dot_e1 / (c00 - (e0_dot_e2 * e1_dot_e2))

        sol = numpy.column_stack((a, b, c)) * cell_volumes[:, None]

        return cell_volumes, sol

    def compute_triangle_circumcenters(self, X):
        '''Computes the center of the circumcenter of all given triangles
        '''
        # https://en.wikipedia.org/wiki/Circumscribed_circle#Higher_dimensions
        a = X[:, 0, :] - X[:, 2, :]
        b = X[:, 1, :] - X[:, 2, :]
        a_dot_a = _row_dot(a, a)
        a2_b = b * a_dot_a[..., None]
        b_dot_b = _row_dot(b, b)
        b2_a = a * b_dot_b[..., None]
        a_cross_b = numpy.cross(a, b)
        N = numpy.cross(a2_b - b2_a, a_cross_b)
        a_cross_b2 = _row_dot(a_cross_b, a_cross_b)
        return 0.5 * N / a_cross_b2[..., None] + X[:, 2, :]

    def get_edges(self, subdomain):
        if subdomain not in self.subdomains:
            self.mark_subdomain(subdomain)
        return self.subdomains[subdomain]['edges']

    def get_vertices(self, subdomain):
        if subdomain not in self.subdomains:
            self.mark_subdomain(subdomain)
        return self.subdomains[subdomain]['vertices']

    def mark_subdomain(self, subdomain):
        # find vertices in subdomain
        if subdomain.is_boundary_only:
            nodes = self.get_vertices('boundary')
        else:
            nodes = self.get_vertices('everywhere')

        subdomain_vertices = []
        for vertex_id in nodes:
            if subdomain.is_inside(self.node_coords[vertex_id]):
                subdomain_vertices.append(vertex_id)
        subdomain_vertices = numpy.unique(subdomain_vertices)

        # extract all edges which are completely or half in the subdomain
        if subdomain.is_boundary_only:
            edges = self.get_edges('boundary')
        else:
            edges = self.get_edges('everywhere')

        subdomain_edges = []
        subdomain_split_edges = []
        for edge_id in edges:
            verts = self.edges['nodes'][edge_id]
            if verts[0] in subdomain_vertices:
                if verts[1] in subdomain_vertices:
                    subdomain_edges.append(edge_id)
                else:
                    subdomain_split_edges.append(edge_id)

        subdomain_edges = numpy.unique(subdomain_edges)
        subdomain_split_edges = numpy.unique(subdomain_split_edges)

        name = subdomain.__class__
        self.subdomains[subdomain] = {
                'vertices': subdomain_vertices,
                'edges': subdomain_edges,
                'split_edges': subdomain_split_edges
                }

        return
