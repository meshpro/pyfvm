# -*- coding: utf-8 -*-
#
import numpy
import warnings
from pyfvm.base import _base_mesh

__all__ = ['meshTri']


class meshTri(_base_mesh):
    '''Class for handling triangular meshes.

    .. inheritance-diagram:: meshTri
    '''
    def __init__(self, nodes, cells):
        '''Initialization.
        '''
        super(meshTri, self).__init__(nodes, cells)
        self.edges = None
        num_cells = len(cells)
        self.cells = numpy.empty(
                num_cells,
                dtype=numpy.dtype([('nodes', (int, 3))])
                )
        self.cells['nodes'] = cells
        self.cell_volumes = None
        self.cell_circumcenters = None
        self.control_volumes = None

        self.create_adjacent_entities()

        self.compute_edge_lengths()

        return

    def compute_edge_lengths(self):
        self.edge_lengths = numpy.empty(len(self.edges))

        for k, vertices in enumerate(self.edges['nodes']):
            coord0 = self.node_coords[vertices[0]]
            coord1 = self.node_coords[vertices[1]]
            self.edge_lengths[k] = numpy.linalg.norm(coord0 - coord1)

        return

    def mark_subdomains(self, subdomains):
        # TODO
        return

    def create_cell_volumes(self):
        '''Computes the area of all triangles in the mesh.
        '''
        num_cells = len(self.cells['nodes'])
        self.cell_volumes = numpy.empty(num_cells, dtype=float)
        for cell_id, cell in enumerate(self.cells):
            x0, x1, x2 = self.node_coords[cell['nodes']]
            # edge0 = node0 - node1
            # edge1 = node1 - node2
            # self.cell_volumes[cell_id] = \
            #     0.5 * numpy.linalg.norm(numpy.cross(edge0, edge1))
            # Append a third component.
            from vtk import vtkTriangle
            self.cell_volumes[cell_id] = \
                abs(vtkTriangle.TriangleArea(x0, x1, x2))
        return

    def compute_cell_circumcenters(self):
        '''Computes the center of the circumcenter of each cell.
        '''
        from vtk import vtkTriangle
        num_cells = len(self.cells['nodes'])
        self.cell_circumcenters = numpy.empty(num_cells,
                                              dtype=numpy.dtype((float, 3))
                                              )
        for cell_id, cell in enumerate(self.cells):
            x = self.node_coords[cell['nodes']]
            # Project to 2D, compute circumcenter, get its barycentric
            # coordinates, and project those back to 3D.
            x2d = numpy.empty(3, dtype=numpy.dtype((float, 2)))
            vtkTriangle.ProjectTo2D(
                    x[0], x[1], x[2],
                    x2d[0], x2d[1], x2d[2]
                    )
            cc2d = numpy.empty(2, dtype=float)
            vtkTriangle.Circumcircle(x2d[0], x2d[1], x2d[2], cc2d)
            bary = numpy.empty(3, dtype=float)
            vtkTriangle.BarycentricCoords(cc2d, x2d[0], x2d[1], x2d[2], bary)
            self.cell_circumcenters[cell_id] = \
                bary[0] * x[0] + \
                bary[1] * x[1] + \
                bary[2] * x[2]
        return

    def create_adjacent_entities(self):
        '''Setup edge-node and edge-cell relations.
        '''
        if self.edges is not None:
            return
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
        if subdomain != 'everywhere':
            raise NotImplemented('subdomains not yet implemented')
        return self.edges

    def compute_control_volumes(self, variant='voronoi'):
        '''Compute the control volumes of all nodes in the mesh.
        '''
        if variant == 'voronoi':
            self._compute_voronoi_volumes()
        elif variant == 'voronoi flat':
            self._compute_flat_voronoi_volumes()
        elif variant == 'barycentric':
            self._compute_barycentric_volumes()
        else:
            raise ValueError('Unknown volume variant ''%s''.' % variant)
        return

    def _compute_voronoi_volumes(self):
        from vtk import vtkTriangle
        num_nodes = len(self.node_coords)
        self.control_volumes = numpy.zeros(num_nodes, dtype=float)

        # compute cell circumcenters
        # if self.cell_circumcenters is None:
        #    self.compute_cell_circumcenters()

        if self.edges is None:
            self.create_adjacent_entities()

        # For flat meshes, the control volume contributions on a per-edge
        # basis by computing the distance between the circumcenters
        # of two adjacent cells.
        # If the mesh is not flat, however, this does not work. Hence, compute
        # the control volume contributions for each side in a cell separately.

        num_cells = len(self.cells)
        for cell_id in range(num_cells):
            # Project the triangle to 2D.
            x = self.node_coords[self.cells['nodes'][cell_id]]
            # Project to 2D, compute circumcenter, get its barycentric
            # coordinates, and project those back to 3D.
            x2d = numpy.empty(3, dtype=numpy.dtype((float, 2)))
            vtkTriangle.ProjectTo2D(x[0], x[1], x[2],
                                    x2d[0], x2d[1], x2d[2])
            # Compute circumcenter.
            cc = numpy.empty(2)
            vtkTriangle.Circumcircle(x2d[0], x2d[1], x2d[2], cc)

            for edge_id in self.cells['edges'][cell_id]:
                # Move the system such that one of the two end points is in the
                # origin. Deliberately take x2d[0].
                node_ids = self.edges['nodes'][edge_id]

                # Get the local IDs.
                edge_lid = numpy.nonzero(
                            self.cells['edges'][cell_id] == edge_id
                            )[0][0]
                node_lids = numpy.array([
                  numpy.nonzero(
                      self.cells['nodes'][cell_id] == node_ids[0]
                      )[0][0],
                  numpy.nonzero(
                      self.cells['nodes'][cell_id] == node_ids[1]
                      )[0][0]
                  ])

                # This makes use of the fact that cellsEdges and cellsNodes are
                # coordinated such that in cell #i, the edge cellsEdges[i][k]
                # opposes cellsNodes[i][k].
                other0 = x2d[edge_lid] - x2d[node_lids[0]]

                # Compute edge midpoint.
                edge_midpoint = 0.5 * (x2d[node_lids[0]] + x2d[node_lids[1]]) \
                    - x2d[node_lids[0]]

                cc_tmp = cc - x2d[node_lids[0]]

                # Compute the area of the triangle {node[0], cc,
                # edge_midpoint}.  Gauge the sign with the sign of the area
                # {node[0], other0, edge_midpoint}.

                # Computing the triangle volume like this is called the
                # shoelace formula and can be interpreted as the z-component of
                # the cross-product of other0 and edge_midpoint.
                gauge = other0[0] * edge_midpoint[1] \
                    - other0[1] * edge_midpoint[0]

                self.control_volumes[node_ids] += \
                    numpy.sign(gauge) * 0.5 * (
                            cc_tmp[0] * edge_midpoint[1] -
                            cc_tmp[1] * edge_midpoint[0]
                            )
        # Sanity checks.
        if self.cell_volumes is None:
            self.create_cell_volumes()
        sum_cv = sum(self.control_volumes)
        sum_cells = sum(self.cell_volumes)
        alpha = sum_cv - sum_cells
        if abs(alpha) > 1.0e-9:
            msg = ('Sum of control volumes sum does not coincide with the sum '
                   'of the cell volumes (|cv|-|cells| = %g - %g = %g.'
                   ) % (sum_cv, sum_cells, alpha)
            raise RuntimeError(msg)
        if any(self.control_volumes < 0.0):
            msg = 'Not all control volumes are positive. This is due do ' \
                + 'the triangulation not being Delaunay.'
            warnings.warn(msg)
        return

    def _compute_flat_voronoi_volumes(self):
        num_nodes = len(self.node_coords)
        self.control_volumes = numpy.zeros(num_nodes, dtype=float)

        # compute cell circumcenters
        if self.cell_circumcenters is None:
            self.compute_cell_circumcenters()

        if self.edges is None:
            self.create_adjacent_entities()

        # Compute covolumes and control volumes.
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
            cell0 = self.edges['cells'][edge_id][0]
            # This nonzero construct is an ugly replacement for the nonexisting
            # index() method. (Compare with Python lists.)
            edge_lid = \
                numpy.nonzero(self.cells['edges'][cell0] == edge_id)[0][0]
            # This makes use of the fact that cellsEdges and cellsNodes
            # are coordinated such that in cell #i, the edge cellsEdges[i][k]
            # opposes cellsNodes[i][k].
            other0 = self.node_coords[self.cells['nodes'][cell0][edge_lid]] \
                - node
            node_ids = self.edges['nodes'][edge_id]
            node_coords = self.node_coords[node_ids]
            edge_midpoint = 0.5 * (node_coords[0] + node_coords[1]) \
                - node
            # Computing the triangle volume like this is called the shoelace
            # formula and can be interpreted as the z-component of the
            # cross-product of other0 and edge_midpoint.
            gauge = other0[0] * edge_midpoint[1] \
                - other0[1] * edge_midpoint[0]

            # Get the circumcenters of the adjacent cells.
            cc = self.cell_circumcenters[self.edges['cells'][edge_id]] \
                - node
            if len(cc) == 2:  # interior edge
                self.control_volumes[node_ids] += \
                    numpy.sign(gauge) * 0.5 * (cc[0][0] * cc[1][1] -
                                               cc[0][1] * cc[1][0]
                                               )
            elif len(cc) == 1:  # boundary edge
                self.control_volumes[node_ids] += \
                    numpy.sign(gauge) * 0.5 * (cc[0][0] * edge_midpoint[1] -
                                               cc[0][1] * edge_midpoint[0]
                                               )
            else:
                raise RuntimeError('An edge should have either 1'
                                   ' or two adjacent cells.'
                                   )

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
        if self.cell_volumes is None:
            self.create_cell_volumes()
        num_nodes = len(self.node_coords)
        self.control_volumes = numpy.zeros(num_nodes, dtype=float)
        for k, cell in enumerate(self.cells):
            self.control_volumes[cell['nodes']] += self.cell_volumes[k] / 3.0
        return

    def compute_edge_normals(self):
        '''Compute the edge normals, pointing either in the direction of the
        cell with larger GID (for interior edges), or towards the outside of
        the domain (for boundary edges).

        :returns edge_normals: List of all edge normals.
        :type edge_normals: numpy.ndarray(num_edges, numpy.dtype((float, 2)))
        '''
        num_edges = len(self.edges['nodes'])
        edge_normals = numpy.empty(num_edges, dtype=numpy.dtype((float, 2)))
        for cell_id, cell in enumerate(self.cells):
            # Loop over the local faces.
            for k in range(3):
                edge_id = cell['edges'][k]
                # Compute the normal in the direction of the higher cell ID,
                # or if this is a boundary face, to the outside of the domain.
                neighbor_cell_ids = self.edges['cells'][edge_id]
                if cell_id == neighbor_cell_ids[0]:
                    edge_nodes = self.node_coords[self.edges['nodes'][edge_id]]
                    edge = (edge_nodes[1] - edge_nodes[0])
                    edge_normals[edge_id] = numpy.array([-edge[1], edge[0]])
                    edge_normals[edge_id] /= \
                        numpy.linalg.norm(edge_normals[edge_id])

                    # Make sure the normal points in the outward direction.
                    other_node_id = self.cells['nodes'][cell_id][k]
                    other_node_coords = self.node_coords[other_node_id]
                    if numpy.dot(edge_nodes[0]-other_node_coords,
                                 edge_normals[edge_id]
                                 ) < 0.0:
                        edge_normals[edge_id] *= -1
        return edge_normals

    def compute_gradient(self, u):
        '''Computes an approximation to the gradient :math:`\\nabla u` of a
        given scalar valued function :math:`u`, defined in the node points.
        This is taken from :cite:`NME2187`.
        '''
        num_nodes = len(self.node_coords)
        assert len(u) == num_nodes
        gradient = numpy.zeros((num_nodes, 2), dtype=u.dtype)
        # Compute everything we need.
        if self.edges is None:
            self.create_adjacent_entities()
        if self.control_volumes is None:
            self.compute_control_volumes()
        if self.cell_circumcenters is None:
            self.compute_cell_circumcenters()

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
            edge_coords = self.node_coords[edge['nodes'][1]] \
                - self.node_coords[edge['nodes'][0]]

            # Compute coedge length.
            if len(edge['cells']) == 1:
                # Boundary edge.
                edge_midpoint = 0.5 * (self.node_coords[edge['nodes'][0]] +
                                       self.node_coords[edge['nodes'][1]]
                                       )
                coedge = self.cell_circumcenters[edge['cells'][0]] \
                    - edge_midpoint
                coedge_midpoint = \
                    0.5 * (self.cell_circumcenters[edge['cells'][0]] +
                           edge_midpoint
                           )
            elif len(edge['cells']) == 2:
                # Interior edge.
                coedge = self.cell_circumcenters[edge['cells'][0]] \
                    - self.cell_circumcenters[edge['cells'][1]]
                coedge_midpoint = \
                    0.5 * (self.cell_circumcenters[edge['cells'][0]] +
                           self.cell_circumcenters[edge['cells'][1]]
                           )
            else:
                raise RuntimeError('Edge needs to have either one '
                                   'or two neighbors.'
                                   )

            # Compute the coefficient r for both contributions
            coeffs = numpy.sqrt(numpy.dot(coedge, coedge) /
                                numpy.dot(edge_coords, edge_coords)
                                ) / self.control_volumes[edge['nodes']]

            # Compute R*_{IJ} ((11) in [1]).
            r0 = (coedge_midpoint - self.node_coords[edge['nodes'][0]]) \
                * coeffs[0]
            r1 = (coedge_midpoint - self.node_coords[edge['nodes'][1]]) \
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
            \lim_{A\\to 0} n\cdot curl(F) = |A|^{-1} \int_{dA} F dr;

        see http://en.wikipedia.org/wiki/Curl_(mathematics). Actually, to
        approximate the integral, one would only need the projection of the
        vector field onto the edges at the midpoint of the edges.
        '''
        if self.edges is None:
            self.create_adjacent_entities()
        curl = numpy.zeros((len(self.cells), 3), dtype=vector_field.dtype)
        for edge in self.edges:
            edge_coords = self.node_coords[edge['nodes'][1]] \
                - self.node_coords[edge['nodes'][0]]
            # Calculate A at the edge midpoint.
            A = 0.5 * (vector_field[edge['nodes'][0]] +
                       vector_field[edge['nodes'][1]]
                       )
            curl[edge['cells'], :] += edge_coords * numpy.dot(edge_coords, A)
        return curl

    def check_delaunay(self):
        if self.edges is None:
            self.create_adjacent_entities()
        if self.cell_circumcenters is None:
            self.compute_cell_circumcenters()

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

    def show(self, show_covolumes=True, save_as=None):
        '''Show the mesh using matplotlib.

        :param show_covolumes: If true, show all covolumes of the mesh, too.
        :type show_covolumes: bool, optional
        '''
        if self.edges is None:
            self.create_adjacent_entities()

        import matplotlib.pyplot as plt

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
            if self.cell_circumcenters is None:
                self.compute_cell_circumcenters()
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
        if save_as:
            import matplotlib2tikz
            matplotlib2tikz.save(save_as)
        else:
            plt.show()
        return

    def show_node(self, node_id, show_covolume=True):
        '''Plot the vicinity of a node and its covolume.

        :param node_id: Node ID of the node to be shown.
        :type node_id: int

        :param show_covolume: If true, shows the covolume of the node, too.
        :type show_covolume: bool, optional
        '''
        if self.edges['nodes'] is None:
            self.create_adjacent_entities()

        import matplotlib.pyplot as plt

        fig = plt.figure()
        # ax = fig.gca(projection='3d')
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
            if self.cell_circumcenters is None:
                self.compute_cell_circumcenters()
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
        plt.show()
        return
