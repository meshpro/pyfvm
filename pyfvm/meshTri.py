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

        self.create_adjacent_entities()
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
        self.edge_lengths = numpy.sqrt(numpy.sum(edges**2, axis=1))
        return

    def mark_default_subdomains(self):
        self.subdomains = {}
        self.subdomains['everywhere'] = {
                'vertices': range(len(self.node_coords)),
                'edges': range(len(self.edges)),
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
        self.cell_volumes = 0.5 * numpy.sqrt(numpy.sum(a_cross_b**2, axis=1))
        return

    def compute_cell_circumcenters(self):
        '''Computes the center of the circumcenter of each cell.
        '''
        # https://en.wikipedia.org/wiki/Circumscribed_circle#Higher_dimensions
        X = self.node_coords[self.cells['nodes']]
        a = X[:, 0, :] - X[:, 2, :]
        b = X[:, 1, :] - X[:, 2, :]
        a_dot_a = numpy.sum(a**2, axis=1)
        a2_b = (b.T * a_dot_a).T
        b_dot_b = numpy.sum(b**2, axis=1)
        b2_a = (a.T * b_dot_b).T
        a_cross_b = numpy.cross(a, b)
        N = numpy.cross(a2_b - b2_a, a_cross_b)
        a_cross_b2 = numpy.sum(a_cross_b**2, axis=1)
        self.cell_circumcenters = 0.5 * (N.T / a_cross_b2).T + X[:, 2, :]
        return

    def create_adjacent_entities(self):
        '''Setup edge-node and edge-cell relations.
        '''
        # Get upper bound for number of edges; trim later.
        max_num_edges = 3 * len(self.cells['nodes'])

        dt = numpy.dtype([('nodes', (int, 2)), ('cells', numpy.object)])
        self.edges = numpy.empty(max_num_edges, dtype=dt)
        # To create an array of empty lists, do what's described at
        # http://mail.scipy.org/pipermail/numpy-discussion/2009-November/046566.html
        filler = numpy.frompyfunc(lambda x: list(), 1, 1)
        self.edges['cells'] = filler(self.edges['cells'])

        # Extend the self.cells array by the 'edges' 'keyword'.
        dt = numpy.dtype([('nodes', (int, 3)), ('edges', (int, 3))])
        cells = self.cells['nodes']
        self.cells = numpy.empty(len(cells), dtype=dt)
        self.cells['nodes'] = cells

        # The (sorted) dictionary edges keeps track of how nodes and edges
        # are connected.
        # If  node_edges[(3,4)] == 17  is true, then the nodes (3,4) are
        # connected  by edge 17.
        registered_edges = {}

        new_edge_gid = 0
        # Loop over all elements.
        for cell_id, cell in enumerate(self.cells):
            # We're treating simplices so loop over all combinations of
            # local nodes.
            # Make sure cellNodes are sorted.
            self.cells['nodes'][cell_id] = \
                numpy.sort(self.cells['nodes'][cell_id])
            for k in range(len(cell['nodes'])):
                # Remove the k-th element. This makes sure that the k-th
                # edge is opposite of the k-th node. Useful later in
                # in construction of edge (face) normals.
                indices = tuple(cell['nodes'][:k]) \
                    + tuple(cell['nodes'][k+1:])
                if indices in registered_edges:
                    edge_gid = registered_edges[indices]
                    self.edges[edge_gid]['cells'].append(cell_id)
                    self.cells[cell_id]['edges'][k] = edge_gid
                else:
                    # add edge
                    # The alternative
                    #   self.edges[new_edge_gid]['nodes'] = indices
                    # doesn't work here. Check out
                    # http://projects.scipy.org/numpy/ticket/2068
                    self.edges['nodes'][new_edge_gid] = indices
                    # edge['cells'] is also always ordered.
                    self.edges['cells'][new_edge_gid].append(cell_id)
                    self.cells['edges'][cell_id][k] = new_edge_gid
                    registered_edges[indices] = new_edge_gid
                    new_edge_gid += 1
        # trim edges
        self.edges = self.edges[:new_edge_gid]
        return

    def get_edges(self, subdomain):
        return self.subdomains[subdomain]['edges']

    def get_vertices(self, subdomain):
        return self.subdomains[subdomain]['vertices']

    def compute_control_volumes(self, variant='voronoi'):
        '''Compute the control volumes of all nodes in the mesh.
        '''
        if variant == 'voronoi':
            if (abs(self.node_coords[:, 2]) < 1.0e-10).all():
                self._compute_flat_voronoi_volumes()
            else:
                self._compute_voronoi_volumes()
        elif variant == 'barycentric':
            self._compute_barycentric_volumes()
        else:
            raise ValueError('Unknown volume variant ''%s''.' % variant)
        return

    def _edge_comp(self, coords, other_coord, cc):
        # Move the system such that one of the two end points is in the
        # origin. Deliberately take x2d[0].

        other0 = other_coord - coords[0]

        # Compute edge midpoint.
        # edge_midpoint = 0.5 * (coords[0] + coords[1]) - coords[0]
        edge_midpoint = 0.5 * (coords[1] - coords[0])

        cc_tmp = cc - coords[0]

        # Compute the area of the triangle {node[0], cc, edge_midpoint}. Gauge
        # with the sign of the area {node[0], other0, edge_midpoint}.

        # Computing the triangle volume like this is called the shoelace
        # formula and can be interpreted as the z-component of the
        # cross-product of other0 and edge_midpoint.
        val = 0.5 * (
                cc_tmp[0] * edge_midpoint[1] -
                cc_tmp[1] * edge_midpoint[0]
                )
        if other0[0] * edge_midpoint[1] > other0[1] * edge_midpoint[0]:
            return val
        else:
            return -val

    def _compute_voronoi_volumes(self):
        # For flat meshes, the control volume contributions on a per-edge basis
        # by computing the distance between the circumcenters of two adjacent
        # cells. If the mesh is not flat, however, this does not work. Hence,
        # compute the control volume contributions for each side in a cell
        # separately.
        self.control_volumes = numpy.zeros(len(self.node_coords), dtype=float)
        for node_ids in self.cells['nodes']:
            # Project to 2D, compute circumcenter.
            from vtk import vtkTriangle
            x = self.node_coords[node_ids]
            x2d = numpy.empty(3, dtype=numpy.dtype((float, 2)))
            vtkTriangle.ProjectTo2D(
                    x[0], x[1], x[2],
                    x2d[0], x2d[1], x2d[2]
                    )
            # Compute circumcenter.
            cc = numpy.empty(2)
            vtkTriangle.Circumcircle(x2d[0], x2d[1], x2d[2], cc)
            for other_lid in range(3):
                node_lids = range(3)[:other_lid] + range(3)[other_lid+1:]
                my_node_ids = node_ids[node_lids]
                self.control_volumes[my_node_ids] += \
                    self._edge_comp(x2d[node_lids], x2d[other_lid], cc)

        # Sanity checks.
        sum_cv = sum(self.control_volumes)
        sum_cells = sum(self.cell_volumes)
        assert abs(sum_cv - sum_cells) < 1.0e-6

        if any(self.control_volumes < 0.0):
            msg = 'Not all control volumes are positive. This is due do ' \
                + 'the triangulation not being Delaunay.'
            warnings.warn(msg)
        return

    def _compute_flat_voronoi_volumes(self):
        num_nodes = len(self.node_coords)
        self.control_volumes = numpy.zeros(num_nodes, dtype=float)

        num_edges = len(self.edges['nodes'])
        for edge_id in range(num_edges):
            # Move the system such that one of the two end points is in the
            # origin. Deliberately take self.edges['nodes'][edge_id][0].
            node = self.node_coords[self.edges['nodes'][edge_id][0]]

            # The orientation of the coedge needs gauging.  Do it in such as a
            # way that the control volume contribution is positive if and only
            # if the area of the triangle (node, other0, edge_midpoint) (in
            # this order) is positive.  Equivalently, the triangles (node,
            # edge_midpoint, other1) or (node, other0, other1) could  be
            # considered.  other{0,1} refers to the the node opposing the edge
            # in the adjacent cell {0,1}.
            # Get the opposing node of the first adjacent cell.
            cells = self.edges['cells'][edge_id]
            # This nonzero construct is an ugly replacement for the nonexisting
            # index() method. (Compare with Python lists.)
            edge_lid = self.cells['edges'][cells[0]].tolist().index(edge_id)
            # This makes use of the fact that cellsEdges and cellsNodes
            # are coordinated such that in cell #i, the edge cellsEdges[i][k]
            # opposes cellsNodes[i][k].
            other0 = self.node_coords[self.cells['nodes'][cells[0]][edge_lid]] \
                - node
            node_ids = self.edges['nodes'][edge_id]
            node_coords = self.node_coords[node_ids]
            edge_midpoint = 0.5 * (node_coords[0] + node_coords[1]) - node
            # Computing the triangle volume like this is called the shoelace
            # formula and can be interpreted as the z-component of the
            # cross-product of other0 and edge_midpoint.
            positive_gauge = (
                    other0[0] * edge_midpoint[1] > other0[1] * edge_midpoint[0]
                    )

            # Get the circumcenters of the adjacent cells.
            cc = self.cell_circumcenters[self.edges['cells'][edge_id]] \
                - node
            if len(cc) == 2:  # interior edge
                val = 0.5 * (cc[0][0] * cc[1][1] - cc[0][1] * cc[1][0])
            elif len(cc) == 1:  # boundary edge
                val = 0.5 * (
                        cc[0][0] * edge_midpoint[1] -
                        cc[0][1] * edge_midpoint[0]
                        )
            else:
                raise RuntimeError('An edge should have either 1'
                                   ' or two adjacent cells.'
                                   )
            if positive_gauge:
                self.control_volumes[node_ids] += val
            else:
                self.control_volumes[node_ids] -= val

        # Don't check equality with sum of triangle areas since
        # it will generally not be equal.
        if any(self.control_volumes < 0.0):
            msg = 'Not all control volumes are positive. This is due do ' \
                + 'the triangulation not being Delaunay.'
            warnings.warn(msg)
        return

    def _compute_barycentric_volumes(self):
        '''Control volumes based on barycentric splitting.'''
        # The barycentric midpoint "divides the triangle" into three areas of
        # equal volume. Hence, just assign one third of the volumes to the
        # corner points of each cell.
        num_nodes = len(self.node_coords)
        self.control_volumes = numpy.zeros(num_nodes, dtype=float)
        for k, cell in enumerate(self.cells):
            self.control_volumes[cell['nodes']] += self.cell_volumes[k] / 3.0
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
        assert (abs(self.node_coords[:, 2]) < 1.0e-10).all()
        node_coords2d = self.node_coords[:, :2]

        assert (abs(self.cell_circumcenters[:, 2]) < 1.0e-10).all()
        cell_circumcenters2d = self.cell_circumcenters[:, :2]

        num_nodes = len(node_coords2d)
        assert len(u) == num_nodes
        # This only works for flat meshes.

        gradient = numpy.zeros((num_nodes, 2), dtype=u.dtype)

        # Create an empty 2x2 matrix for the boundary nodes to hold the
        # edge correction ((17) in [1]).
        boundary_matrices = {}
        for edge_id, edge in enumerate(self.edges):
            if len(edge['cells']) == 1:
                if edge['nodes'][0] not in boundary_matrices:
                    boundary_matrices[edge['nodes'][0]] = numpy.zeros((2, 2))
                if edge['nodes'][1] not in boundary_matrices:
                    boundary_matrices[edge['nodes'][1]] = numpy.zeros((2, 2))

        for edge_id, edge in enumerate(self.edges):
            # Compute edge length.
            edge_coords = node_coords2d[edge['nodes'][1]] -\
                          node_coords2d[edge['nodes'][0]]

            # Compute coedge length.
            if len(edge['cells']) == 1:
                # Boundary edge.
                edge_midpoint = 0.5 * (
                        node_coords2d[edge['nodes'][0]] +
                        node_coords2d[edge['nodes'][1]]
                        )
                coedge = \
                    cell_circumcenters2d[edge['cells'][0]] - edge_midpoint
                coedge_midpoint = 0.5 * (
                        cell_circumcenters2d[edge['cells'][0]] +
                        edge_midpoint
                        )
            elif len(edge['cells']) == 2:
                # Interior edge.
                coedge = cell_circumcenters2d[edge['cells'][0]] - \
                         cell_circumcenters2d[edge['cells'][1]]
                coedge_midpoint = 0.5 * (
                        cell_circumcenters2d[edge['cells'][0]] +
                        cell_circumcenters2d[edge['cells'][1]]
                        )
            else:
                raise RuntimeError(
                        'Edge needs to have either one or two neighbors.'
                        )

            # Compute the coefficient r for both contributions
            coeffs = numpy.sqrt(numpy.dot(coedge, coedge) /
                                numpy.dot(edge_coords, edge_coords)
                                ) / self.control_volumes[edge['nodes']]

            # Compute R*_{IJ} ((11) in [1]).
            r0 = (coedge_midpoint - node_coords2d[edge['nodes'][0]]) \
                * coeffs[0]
            r1 = (coedge_midpoint - node_coords2d[edge['nodes'][1]]) \
                * coeffs[1]

            diff = u[edge['nodes'][1]] - u[edge['nodes'][0]]

            gradient[edge['nodes'][0]] += r0 * diff
            gradient[edge['nodes'][1]] -= r1 * diff

            # Store the boundary correction matrices.
            if edge['nodes'][0] in boundary_matrices:
                boundary_matrices[edge['nodes'][0]] += \
                    numpy.outer(r0, edge_coords)
            if edge['nodes'][1] in boundary_matrices:
                boundary_matrices[edge['nodes'][1]] += \
                    numpy.outer(r1, -edge_coords)

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
        curl = numpy.zeros(
            (len(self.cells), 3),
            dtype=vector_field.dtype
            )
        for edge in self.edges:
            x0 = self.node_coords[edge['nodes'][0]]
            x1 = self.node_coords[edge['nodes'][1]]
            edge_coords = x1 - x0
            # Compute A at the edge midpoint.
            A = 0.5 * (vector_field[edge['nodes'][0]] +
                       vector_field[edge['nodes'][1]]
                       )
            for k in edge['cells']:
                center = 1./3. * sum(self.node_coords[self.cells['nodes'][k]])
                direction = numpy.cross(x0 - center, x1 - center)
                direction /= numpy.linalg.norm(direction)
                curl[k, :] += direction * numpy.dot(edge_coords, A)
        for k in range(len(curl)):
            curl[k, :] /= self.cell_volumes[k]
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
            covolume_col = '0.6'
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
        A = numpy.array([
            numpy.dot(edges[cell['edges']], edges[cell['edges']].T)
            for cell in self.cells
            ])

        # TODO Perhaps simply perform the dot product here again?
        rhs = numpy.array([
            self.cell_volumes[k] *
            numpy.array([A[k, 0, 0], A[k, 1, 1], A[k, 2, 2]])
            for k in range(len(self.cells))
            ])
        A = A**2

        # Solve all 3x3 systems at once ("broadcasted").
        # If the matrix A is (close to) singular if and only if the cell is
        # (close to being) degenerate. Hence, it has volume 0, and so all the
        # edge coefficients are 0, too. Hence, do nothing.
        sol = numpy.linalg.solve(A, rhs)

        num_edges = len(self.edges)
        self.covolumes = numpy.zeros(num_edges, dtype=float)
        for k, cell in enumerate(self.cells):
            cell_edge_gids = cell['edges']
            self.covolumes[cell_edge_gids] += sol[k]

        # Here, self.covolumes contains the covolume-edgelength ratios. Make
        # sure we end up with the covolumes.
        self.covolumes *= self.edge_lengths

        return

    def compute_surface_areas(self):
        # loop over all boundary edges
        self.surface_areas = numpy.zeros(len(self.get_vertices('everywhere')))
        for edge in self.get_edges('Boundary'):
            vertex0 = self.edges['nodes'][edge][0]
            self.surface_areas[vertex0] += 0.5 * self.edge_lengths[edge]
            vertex1 = self.edges['nodes'][edge][1]
            self.surface_areas[vertex1] += 0.5 * self.edge_lengths[edge]
        return
