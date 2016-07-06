# -*- coding: utf-8 -*-
#
import numpy
import warnings
from pyfvm.base import _base_mesh
import matplotlib as mpl
import os
if 'DISPLAY' not in os.environ:
    # headless mode, for remote executions (and travis)
    mpl.use('Agg')
from matplotlib import pyplot as plt

__all__ = ['meshTri']


def _row_dot(a, b):
    # http://stackoverflow.com/a/26168677/353337
    return numpy.einsum('ij, ij->i', a, b)


class meshTri(_base_mesh):
    '''Class for handling triangular meshes.

    .. inheritance-diagram:: meshTri
    '''
    def __init__(self, nodes, cells):
        '''Initialization.
        '''
        super(meshTri, self).__init__(nodes, cells)
        self.cells = numpy.empty(
                len(cells),
                dtype=numpy.dtype([('nodes', (int, 3))])
                )
        self.cells['nodes'] = cells
        self.create_cell_volumes()

        self.create_edges()
        # self.create_halfedges()
        self.compute_cell_circumcenters()
        self.compute_control_volumes()
        self.compute_edge_lengths()
        self.compute_covolumes()

        self.mark_default_subdomains()

        self.compute_surface_areas()

        return

    def compute_edge_lengths(self):
        edges = self.node_coords[self.edges['nodes'][:, 1]] \
            - self.node_coords[self.edges['nodes'][:, 0]]
        self.edge_lengths = numpy.sqrt(_row_dot(edges, edges))
        return

    def mark_default_subdomains(self):
        self.subdomains = {}
        self.subdomains['everywhere'] = {
                'vertices': range(len(self.node_coords)),
                'edges': range(len(self.edges['nodes'])),
                'half_edges': []
                }

        # Find the boundary edges, i.e., all edges that belong to just one
        # cell.
        boundary_edges = []
        for k, edge_cells in enumerate(self.edges['cells']):
            if len(edge_cells) == 1:
                boundary_edges.append(k)

        # Get vertices on the boundary edges
        boundary_vertices = numpy.unique(
                self.edges['nodes'][boundary_edges].flatten()
                )

        self.subdomains['Boundary'] = {
                'vertices': boundary_vertices,
                'edges': boundary_edges,
                'half_edges': []
                }

        return

    def mark_subdomains(self, subdomains):
        for subdomain in subdomains:
            # find vertices in subdomain
            if subdomain.is_boundary_only:
                nodes = self.get_vertices('Boundary')
            else:
                nodes = self.get_vertices('everywhere')

            subdomain_vertices = []
            for vertex_id in nodes:
                if subdomain.is_inside(self.node_coords[vertex_id]):
                    subdomain_vertices.append(vertex_id)
            subdomain_vertices = numpy.unique(subdomain_vertices)

            # extract all edges which are completely or half in the subdomain
            if subdomain.is_boundary_only:
                edges = self.get_edges('Boundary')
            else:
                edges = self.get_edges('everywhere')

            subdomain_edges = []
            subdomain_half_edges = []
            for edge_id in edges:
                verts = self.edges['nodes'][edge_id]
                if verts[0] in subdomain_vertices:
                    if verts[1] in subdomain_vertices:
                        subdomain_edges.append(edge_id)
                    else:
                        subdomain_half_edges.append(edge_id)

            subdomain_edges = numpy.unique(subdomain_edges)
            subdomain_half_edges = numpy.unique(subdomain_half_edges)

            name = subdomain.__class__.__name__
            self.subdomains[name] = {
                    'vertices': subdomain_vertices,
                    'edges': subdomain_edges,
                    'half_edges': subdomain_half_edges
                    }

        return

    def create_cell_volumes(self):
        '''Computes the area of all triangles in the mesh.
        '''
        X = self.node_coords[self.cells['nodes']]
        a = X[:, 0, :] - X[:, 2, :]
        b = X[:, 1, :] - X[:, 2, :]
        a_cross_b = numpy.cross(a, b)
        self.cell_volumes = 0.5 * numpy.sqrt(_row_dot(a_cross_b, a_cross_b))
        return

    def compute_cell_circumcenters(self):
        '''Computes the center of the circumcenter of each cell.
        '''
        # https://en.wikipedia.org/wiki/Circumscribed_circle#Higher_dimensions
        X = self.node_coords[self.cells['nodes']]
        a = X[:, 0, :] - X[:, 2, :]
        b = X[:, 1, :] - X[:, 2, :]
        a_dot_a = _row_dot(a, a)
        a2_b = b * a_dot_a[..., None]
        b_dot_b = _row_dot(b, b)
        b2_a = a * b_dot_b[..., None]
        a_cross_b = numpy.cross(a, b)
        N = numpy.cross(a2_b - b2_a, a_cross_b)
        a_cross_b2 = _row_dot(a_cross_b, a_cross_b)
        self.cell_circumcenters = 0.5 * N / a_cross_b2[..., None] + X[:, 2, :]
        return

    def create_edges(self):
        '''Setup edge-node and edge-cell relations.
        '''
        self.cells['nodes'].sort(axis=1)
        a = numpy.vstack([
            self.cells['nodes'][:, [0, 1]],
            self.cells['nodes'][:, [1, 2]],
            self.cells['nodes'][:, [0, 2]]
            ])

        # Find the unique edges
        b = numpy.ascontiguousarray(a).view(
                numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
                )
        _, idx, inv = numpy.unique(b, return_index=True, return_inverse=True)
        edge_nodes = a[idx]

        # Create edge->cells relationships
        num_cells = len(self.cells['nodes'])
        edge_cells = [[] for k in range(len(idx))]
        for k, edge_id in enumerate(inv):
            edge_cells[edge_id].append(k % num_cells)

        self.edges = {
            'nodes': edge_nodes,
            'cells': edge_cells
            }

        # cell->edges relationship
        cells_edges = inv.reshape([3, num_cells]).T
        cells = self.cells['nodes']
        self.cells = {
            'nodes': cells,
            'edges': cells_edges
            }

        return

    def get_edges(self, subdomain):
        return self.subdomains[subdomain]['edges']

    def get_vertices(self, subdomain):
        return self.subdomains[subdomain]['vertices']

    def compute_control_volumes(self):
        '''Compute the control volumes of all nodes in the mesh.
        '''
        # For flat meshes, the control volume contributions on a per-edge basis
        # by computing the distance between the circumcenters of two adjacent
        # cells. If the mesh is not flat, however, this does not work. Hence,
        # compute the control volume contributions for each side in a cell
        # separately.
        self.control_volumes = numpy.zeros(len(self.node_coords), dtype=float)
        X = self.node_coords[self.cells['nodes']]
        for other_lid in range(3):
            node_lids = range(3)[:other_lid] + range(3)[other_lid+1:]

            edge_coords = X[:, node_lids, :]
            other_coord = X[:, other_lid, :]

            # Move the system such that one of the two end points is in the
            # origin. Deliberately take coords[0].
            other0 = other_coord - edge_coords[:, 0, :]
            cc = self.cell_circumcenters - edge_coords[:, 0, :]
            # edge_midpoint = 0.5 * (coords[0] + coords[1]) - coords[0]
            edge_midpoints = 0.5 * (
                edge_coords[:, 1, :] - edge_coords[:, 0, :]
                )

            # Compute the area of the triangle {node[0], cc, edge_midpoint}.
            # Gauge with the sign of the area {node[0], other0, edge_midpoint}.
            V = 0.5 * numpy.cross(cc, edge_midpoints)
            # Get normalized gauge vector
            gauge = numpy.cross(other0, edge_midpoints)
            gauge_norm = numpy.sqrt(_row_dot(gauge, gauge))
            gauge /= gauge_norm[..., None]

            val = _row_dot(V, gauge)

            # Add edge contributions into the vertex values of control volumes.
            my_node_ids = self.cells['nodes'][:, node_lids]
            numpy.add.at(self.control_volumes, my_node_ids[:, 0], val)
            numpy.add.at(self.control_volumes, my_node_ids[:, 1], val)

        # Sanity checks.
        sum_cv = sum(self.control_volumes)
        sum_cells = sum(self.cell_volumes)
        assert abs(sum_cv - sum_cells) < 1.0e-6

        if any(self.control_volumes < 0.0):
            msg = 'Not all control volumes are positive. This is due do ' \
                + 'the triangulation not being Delaunay.'
            warnings.warn(msg)
        return

    def compute_gradient(self, u):
        '''Computes an approximation to the gradient :math:`\\nabla u` of a
        given scalar valued function :math:`u`, defined in the node points.
        This is taken from :cite:`NME2187`,

           Discrete gradient method in solid mechanics,
           Lu, Jia and Qian, Jing and Han, Weimin,
           International Journal for Numerical Methods in Engineering,
           http://dx.doi.org/10.1002/nme.2187.
        '''
        # This only works for flat meshes.
        assert (abs(self.node_coords[:, 2]) < 1.0e-10).all()
        node_coords2d = self.node_coords[:, :2]
        cell_circumcenters2d = self.cell_circumcenters[:, :2]

        num_nodes = len(node_coords2d)
        assert len(u) == num_nodes

        gradient = numpy.zeros((num_nodes, 2), dtype=u.dtype)

        # Create an empty 2x2 matrix for the boundary nodes to hold the
        # edge correction ((17) in [1]).
        boundary_matrices = {}
        for node in self.get_vertices('Boundary'):
            boundary_matrices[node] = numpy.zeros((2, 2))

        for edge_id, edge in enumerate(self.edges['cells']):
            # Compute edge length.
            node0 = self.edges['nodes'][edge_id][0]
            node1 = self.edges['nodes'][edge_id][1]

            # Compute coedge length.
            if len(self.edges['cells'][edge_id]) == 1:
                # Boundary edge.
                edge_midpoint = 0.5 * (
                        node_coords2d[node0] +
                        node_coords2d[node1]
                        )
                cell0 = self.edges['cells'][edge_id][0]
                coedge = cell_circumcenters2d[cell0] - edge_midpoint
                coedge_midpoint = 0.5 * (
                        cell_circumcenters2d[cell0] +
                        edge_midpoint
                        )
            elif len(self.edges['cells'][edge_id]) == 2:
                cell0 = self.edges['cells'][edge_id][0]
                cell1 = self.edges['cells'][edge_id][1]
                # Interior edge.
                coedge = cell_circumcenters2d[cell0] - \
                    cell_circumcenters2d[cell1]
                coedge_midpoint = 0.5 * (
                        cell_circumcenters2d[cell0] +
                        cell_circumcenters2d[cell1]
                        )
            else:
                raise RuntimeError(
                        'Edge needs to have either one or two neighbors.'
                        )

            # Compute the coefficient r for both contributions
            # The term
            #    numpy.sqrt(numpy.dot(coedge, coedge))
            # could be replaced by (self.covolumes[edge_id]). However, the two
            # values are always slightly off (1.0e-6 or so). That shouldn't be
            # the case, actually.
            coeffs = numpy.sqrt(numpy.dot(coedge, coedge)) / \
                self.edge_lengths[edge_id] / \
                self.control_volumes[self.edges['nodes'][edge_id]]

            # Compute R*_{IJ} ((11) in [1]).
            r0 = (coedge_midpoint - node_coords2d[node0]) * coeffs[0]
            r1 = (coedge_midpoint - node_coords2d[node1]) * coeffs[1]

            diff = u[node1] - u[node0]

            gradient[node0] += r0 * diff
            gradient[node1] -= r1 * diff

            # Store the boundary correction matrices.
            edge_coords = node_coords2d[node1] - node_coords2d[node0]
            if node0 in boundary_matrices:
                boundary_matrices[node0] += numpy.outer(r0, edge_coords)
            if node1 in boundary_matrices:
                boundary_matrices[node1] += numpy.outer(r1, -edge_coords)

        # Apply corrections to the gradients on the boundary.
        for k, value in boundary_matrices.items():
            gradient[k] = numpy.linalg.solve(value, gradient[k])

        return gradient

    def compute_curl(self, vector_field):
        '''Computes the curl of a vector field. While the vector field is
        point-based, the curl will be cell-based. The approximation is based on

        .. math::
            n\cdot curl(F) = \lim_{A\\to 0} |A|^{-1} \int_{dGamma} F dr;

        see <https://en.wikipedia.org/wiki/Curl_(mathematics)>. Actually, to
        approximate the integral, one would only need the projection of the
        vector field onto the edges at the midpoint of the edges.
        '''
        edge_coords = \
            self.node_coords[self.edges['nodes'][:, 1]] - \
            self.node_coords[self.edges['nodes'][:, 0]]

        barycenters = 1./3. * numpy.sum(
                self.node_coords[self.cells['nodes']],
                axis=1
                )

        # Compute the projection of A on the edge at each edge midpoint.
        nodes = self.edges['nodes']
        x = self.node_coords[nodes]
        A = 0.5 * numpy.sum(vector_field[nodes], axis=1)
        edge_dot_A = _row_dot(edge_coords, A)

        directions = numpy.cross(
                x[self.cells['edges'], 0] - barycenters[:, None, :],
                x[self.cells['edges'], 1] - barycenters[:, None, :]
                )
        dir_nrms = numpy.sqrt(numpy.sum(directions**2, axis=2))
        directions /= dir_nrms[..., None]

        # a: directions scaled with edge_dot_a
        a = directions * edge_dot_A[self.cells['edges']][..., None]

        # sum over all local edges
        curl = numpy.sum(a, axis=1)
        # Divide by cell volumes
        curl /= self.cell_volumes[..., None]
        return curl

    def check_delaunay(self):
        num_interior_edges = 0
        num_delaunay_violations = 0

        num_edges = len(self.edges['nodes'])
        for edge_id in range(num_edges):
            # Boundary edges don't need to be checked.
            if len(self.edges['cells'][edge_id]) != 2:
                continue

            num_interior_edges += 1

            # Each interior edge divides the domain into to half-planes.  The
            # Delaunay condition is fulfilled if and only if the circumcenters
            # of the adjacent cells are in "the right order", i.e., line
            # between the nodes of the cells which do not sit on the hyperplane
            # have the same orientation as the line between the circumcenters.

            # The orientation of the coedge needs gauging.  Do it in such as a
            # way that the control volume contribution is positive if and only
            # if the area of the triangle (node, other0, edge_midpoint) (in
            # this order) is positive.  Equivalently, the triangles (node,
            # edge_midpoint, other1) or (node, other0, other1) could  be
            # considered.  other{0,1} refers to the the node opposing the edge
            # in the adjacent cell {0,1}.
            # Get the opposing node of the first adjacent cell.
            cell0 = self.edges['cells'][edge_id][0]
            # This nonzero construct is an ugly replacement for the nonexisting
            # index() method. (Compare with Python lists.)
            edge_lid = numpy.nonzero(
                    self.cells['edges'][cell0] == edge_id
                    )[0][0]
            # This makes use of the fact that cellsEdges and cellsNodes
            # are coordinated such that in cell #i, the edge cellsEdges[i][k]
            # opposes cellsNodes[i][k].
            other0 = self.node_coords[self.cells['nodes'][cell0][edge_lid]]

            # Get the edge midpoint.
            node_ids = self.edges['nodes'][edge_id]
            node_coords = self.node_coords[node_ids]
            edge_midpoint = 0.5 * (node_coords[0] + node_coords[1])

            # Get the circumcenters of the adjacent cells.
            cc = self.cell_circumcenters[self.edges['cells'][edge_id]]
            # Check if cc[1]-cc[0] and the gauge point
            # in the "same" direction.
            if numpy.dot(edge_midpoint-other0, cc[1]-cc[0]) < 0.0:
                num_delaunay_violations += 1
        return num_delaunay_violations, num_interior_edges

    def show(self, show_covolumes=True):
        '''Show the mesh using matplotlib.

        :param show_covolumes: If true, show all covolumes of the mesh, too.
        :type show_covolumes: bool, optional
        '''
        # from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.gca()
        plt.axis('equal')

        # plot edges
        col = 'k'
        for node_ids in self.edges['nodes']:
            x = self.node_coords[node_ids]
            ax.plot(x[:, 0], x[:, 1], col)

        # Highlight covolumes.
        if show_covolumes:
            covolume_col = '0.8'
            for edge_id in range(len(self.edges['cells'])):
                ccs = self.cell_circumcenters[self.edges['cells'][edge_id]]
                if len(ccs) == 2:
                    p = ccs.T
                elif len(ccs) == 1:
                    edge_midpoint = \
                        0.5 * (
                            self.node_coords[self.edges['nodes'][edge_id][0]] +
                            self.node_coords[self.edges['nodes'][edge_id][1]]
                            )
                    p = numpy.c_[ccs[0], edge_midpoint]
                else:
                    raise RuntimeError('An edge has to have either 1 '
                                       'or 2 adjacent cells.'
                                       )
                ax.plot(p[0], p[1], color=covolume_col)
        return

    def show_node(self, node_id, show_covolume=True):
        '''Plot the vicinity of a node and its covolume.

        :param node_id: Node ID of the node to be shown.
        :type node_id: int

        :param show_covolume: If true, shows the covolume of the node, too.
        :type show_covolume: bool, optional
        '''
        fig = plt.figure()
        ax = fig.gca()
        plt.axis('equal')

        # plot edges
        col = 'k'
        for node_ids in self.edges['nodes']:
            if node_id in node_ids:
                x = self.node_coords[node_ids]
                ax.plot(x[:, 0],
                        x[:, 1],
                        col)

        # Highlight covolumes.
        if show_covolume:
            covolume_boundary_col = '0.5'
            covolume_area_col = '0.7'
            for edge_id in range(len(self.edges['cells'])):
                node_ids = self.edges['nodes'][edge_id]
                if node_id in node_ids:
                    ccs = self.cell_circumcenters[self.edges['cells'][edge_id]]
                    if len(ccs) == 2:
                        p = ccs.T
                        q = numpy.c_[ccs[0], ccs[1], self.node_coords[node_id]]
                    elif len(ccs) == 1:
                        edge_midpoint = 0.5 * (
                                self.node_coords[node_ids[0]] +
                                self.node_coords[node_ids[1]]
                                )
                        p = numpy.c_[ccs[0], edge_midpoint]
                        q = numpy.c_[
                                ccs[0],
                                edge_midpoint,
                                self.node_coords[node_id]
                                ]
                    else:
                        raise RuntimeError('An edge has to have either 1 or 2'
                                           'adjacent cells.'
                                           )
                    ax.fill(q[0], q[1], color=covolume_area_col)
                    ax.plot(p[0], p[1], color=covolume_boundary_col)
        return

    def compute_covolumes(self):
        # Precompute edges.
        edges = \
            self.node_coords[self.edges['nodes'][:, 1]] - \
            self.node_coords[self.edges['nodes'][:, 0]]

        # Build the equation system:
        # The equation
        #
        # |simplex| ||u||^2 = \sum_i \alpha_i <u,e_i> <e_i,u>
        #
        # has to hold for all vectors u in the plane spanned by the edges,
        # particularly by the edges themselves.
        cells_edges = edges[self.cells['edges']]
        # <http://stackoverflow.com/a/38110345/353337>
        A = numpy.einsum('ijk,ilk->ijl', cells_edges, cells_edges)
        A = A**2

        # Compute the RHS  cell_volume * <edge, edge>.
        # The dot product <edge, edge> is also on the diagonals of A (before
        # squaring), but simply computing it again is cheaper than extracting
        # it from A.
        edge_dot_edge = _row_dot(edges, edges)
        rhs = edge_dot_edge[self.cells['edges']] * self.cell_volumes[..., None]

        # Solve all 3x3 systems at once ("broadcast").
        # If the matrix A is (close to) singular if and only if the cell is
        # (close to being) degenerate. Hence, it has volume 0, and so all the
        # edge coefficients are 0, too. Hence, do nothing.
        sol = numpy.linalg.solve(A, rhs)

        num_edges = len(self.edges['nodes'])
        self.covolumes = numpy.zeros(num_edges, dtype=float)
        numpy.add.at(
                self.covolumes,
                self.cells['edges'].flatten(),
                sol.flatten()
                )

        # Here, self.covolumes contains the covolume-edgelength ratios. Make
        # sure we end up with the covolumes.
        self.covolumes *= self.edge_lengths

        return

    def compute_surface_areas(self):
        self.surface_areas = numpy.zeros(len(self.get_vertices('everywhere')))
        b_edge = self.get_edges('Boundary')
        numpy.add.at(
            self.surface_areas,
            self.edges['nodes'][b_edge, 0],
            0.5 * self.edge_lengths[b_edge]
            )
        numpy.add.at(
            self.surface_areas,
            self.edges['nodes'][b_edge, 1],
            0.5 * self.edge_lengths[b_edge]
            )
        return
