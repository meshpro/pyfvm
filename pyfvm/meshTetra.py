# -*- coding: utf-8 -*-
#
import numpy
from pyfvm.base import _base_mesh, _row_dot

__all__ = ['meshTetra']


def _my_dot(a, b):
    return numpy.einsum('ijk, ijk->ij', a, b)


class meshTetra(_base_mesh):
    '''Class for handling tetrahedral meshes.

    .. inheritance-diagram:: meshTetra
    '''
    def __init__(self, node_coords, cells, mode='algebraic'):
        '''Initialization.
        '''
        # Make sure all nodes appear in the cells.
        cell_nodes = numpy.unique(cells.flatten())
        assert len(cell_nodes) == len(nodes)

        super(meshTetra, self).__init__(node_coords, cells)

        self.cells = {
            'nodes': cells
            }

        self.create_adjacent_entities()
        self.create_cell_circumcenters_and_volumes()
        self.compute_edge_lengths()
        if mode == 'geometric':
            self.compute_ce_ratios_geometric()
        elif mode == 'algebraic':
            self.compute_ce_ratios_algebraic()
        else:
            raise ValueError('Illegal mode \'%s\'.' % mode)

        self.compute_control_volumes()

        self.mark_default_subdomains()
        return

    def mark_default_subdomains(self):
        self.subdomains = {}
        self.subdomains['everywhere'] = {
                'vertices': range(len(self.node_coords)),
                'edges': range(len(self.edges['nodes'])),
                'faces': range(len(self.faces['nodes']))
                }

        # Get vertices on the boundary faces
        boundary_faces = numpy.where(self.is_boundary_face)[0]
        boundary_vertices = numpy.unique(
                self.faces['nodes'][boundary_faces].flatten()
                )

        self.subdomains['Boundary'] = {
                'vertices': boundary_vertices
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

            name = subdomain.__class__.__name__
            self.subdomains[name] = {
                    'vertices': subdomain_vertices
                    }

        return

    def get_edges(self, subdomain):
        return self.subdomains[subdomain]['edges']

    def get_vertices(self, subdomain):
        return self.subdomains[subdomain]['vertices']

    def create_adjacent_entities(self):
        '''Set up edge-node, edge-cell, edge-face, face-node, and face-cell
        relations.
        '''
        self.cells['nodes'].sort(axis=1)

        self.create_cell_face_relationships()
        self.create_face_edge_relationships()

        return

    def create_cell_face_relationships(self):
        # All possible faces
        a = numpy.vstack([
            self.cells['nodes'][:, [0, 1, 2]],
            self.cells['nodes'][:, [0, 1, 3]],
            self.cells['nodes'][:, [0, 2, 3]],
            self.cells['nodes'][:, [1, 2, 3]]
            ])
        other = numpy.vstack([
            self.cells['nodes'][:, [3]],
            self.cells['nodes'][:, [2]],
            self.cells['nodes'][:, [1]],
            self.cells['nodes'][:, [0]]
            ])

        # Find the unique faces
        b = numpy.ascontiguousarray(a).view(
                numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
                )
        _, idx, inv, cts = numpy.unique(
                b,
                return_index=True,
                return_inverse=True,
                return_counts=True
                )
        face_nodes = a[idx]

        self.is_boundary_face = (cts == 1)

        self.faces = {
            'nodes': face_nodes
            }

        # cell->faces relationship
        num_cells = len(self.cells['nodes'])
        cells_faces = inv.reshape([4, num_cells]).T
        self.cells['faces'] = cells_faces

        # Store the opposing nodes too
        self.cells['opposing vertex'] = other.reshape([4, num_cells]).T

        # save for create_edge_cells
        self._inv_faces = inv

        return

    def create_face_cells(self):
        # Create face->cells relationships
        num_cells = len(self.cells['nodes'])
        face_cells = [[] for k in range(len(self.faces['nodes']))]
        for k, face_id in enumerate(self._inv_faces):
            face_cells[face_id].append(k % num_cells)
        self.faces['cells'] = face_cells
        return

    def create_face_edge_relationships(self):
        a = numpy.vstack([
            self.faces['nodes'][:, [0, 1]],
            self.faces['nodes'][:, [0, 2]],
            self.faces['nodes'][:, [1, 2]]
            ])

        # Find the unique edges
        b = numpy.ascontiguousarray(a).view(
                numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
                )
        _, idx, inv = numpy.unique(
                b,
                return_index=True,
                return_inverse=True
                )
        edge_nodes = a[idx]

        self.edges = {
            'nodes': edge_nodes
            }

        # face->edge relationship
        num_faces = len(self.faces['nodes'])
        face_edges = inv.reshape([3, num_faces]).T
        self.faces['edges'] = face_edges

        return

    def create_cell_circumcenters_and_volumes(self):
        '''Computes the center of the circumsphere of each cell.
        '''
        cell_coords = self.node_coords[self.cells['nodes']]

        a = cell_coords[:, 1, :] - cell_coords[:, 0, :]
        b = cell_coords[:, 2, :] - cell_coords[:, 0, :]
        c = cell_coords[:, 3, :] - cell_coords[:, 0, :]

        omega = _row_dot(a, numpy.cross(b, c))

        self.cell_circumcenters = cell_coords[:, 0, :] + (
                numpy.cross(b, c) * _row_dot(a, a)[:, None] +
                numpy.cross(c, a) * _row_dot(b, b)[:, None] +
                numpy.cross(a, b) * _row_dot(c, c)[:, None]
                ) / (2.0 * omega[:, None])

        # https://en.wikipedia.org/wiki/Tetrahedron#Volume
        self.cell_volumes = abs(omega) / 6.0
        return

    def compute_ce_ratios_algebraic(self):
        # Precompute edges.
        edges = \
            self.node_coords[self.edges['nodes'][:, 1]] - \
            self.node_coords[self.edges['nodes'][:, 0]]

        # create cells -> edges
        num_cells = len(self.cells['nodes'])
        cells_edges = numpy.empty((num_cells, 6), dtype=int)
        for cell_id, face_ids in enumerate(self.cells['faces']):
            edges_set = set(self.faces['edges'][face_ids].flatten())
            cells_edges[cell_id] = list(edges_set)

        self.cells['edges'] = cells_edges

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

        # Solve all k-by-k systems at once ("broadcast"). (`k` is the number of
        # edges per simplex here.)
        # If the matrix A is (close to) singular if and only if the cell is
        # (close to being) degenerate. Hence, it has volume 0, and so all the
        # edge coefficients are 0, too. Hence, do nothing.
        sol = numpy.linalg.solve(A, rhs)

        num_edges = len(self.edges['nodes'])
        self.ce_ratios = numpy.zeros(num_edges, dtype=float)
        numpy.add.at(
                self.ce_ratios,
                self.cells['edges'],
                sol
                )

        return

    def compute_ce_ratios_geometric(self):

        v0 = self.faces['nodes'][self.cells['faces']][:, :, 0]
        v1 = self.faces['nodes'][self.cells['faces']][:, :, 1]
        v2 = self.faces['nodes'][self.cells['faces']][:, :, 2]
        v_op = self.cells['opposing vertex']

        x0 = self.node_coords[v0] - self.node_coords[v_op]
        x1 = self.node_coords[v1] - self.node_coords[v_op]
        x2 = self.node_coords[v2] - self.node_coords[v_op]

        e0_cross_e1 = numpy.cross(x2 - x0, x1 - x0)
        face_areas = numpy.sqrt(_my_dot(e0_cross_e1, e0_cross_e1))

        x0_cross_x1 = numpy.cross(x0, x1)
        x1_cross_x2 = numpy.cross(x1, x2)
        x2_cross_x0 = numpy.cross(x2, x0)
        x0_dot_x0 = _my_dot(x0, x0)
        x1_dot_x1 = _my_dot(x1, x1)
        x2_dot_x2 = _my_dot(x2, x2)

        a = (
            2 * _my_dot(numpy.cross(x0, x1), x2)**2 -
            _my_dot(
                x0_cross_x1 + x1_cross_x2 + x2_cross_x0,
                x0_cross_x1 * x2_dot_x2[..., None] +
                x1_cross_x2 * x0_dot_x0[..., None] +
                x2_cross_x0 * x1_dot_x1[..., None]
            )) / (12.0 * face_areas)

        # Distances of the cell circumcenter to the faces.
        # (shape: num_cells x 4)
        d = a / self.cell_volumes[:, None]

        # prepare face edges
        e = self.node_coords[self.edges['nodes'][self.faces['edges'], 1]] - \
            self.node_coords[self.edges['nodes'][self.faces['edges'], 0]]

        e0 = e[:, 0, :]
        e1 = e[:, 1, :]
        e2 = e[:, 2, :]

        _, face_ce_ratios = self.compute_tri_areas_and_ce_ratios(e0, e1, e2)

        # Multiply
        s = 0.5 * face_ce_ratios * d[..., None]

        # Add s together for the edges
        self.ce_ratios = numpy.zeros(len(self.edges['nodes']))
        numpy.add.at(
                self.ce_ratios,
                self.faces['edges'][self.cells['faces']],
                s
                )
        return

    def compute_control_volumes(self):
        '''Compute the control volumes of all nodes in the mesh.
        '''
        self.control_volumes = numpy.zeros(len(self.node_coords), dtype=float)

        #   1/3. * (0.5 * edge_length) * covolume
        # = 1/6 * edge_length**2 * ce_ratio_edge_ratio
        e = self.node_coords[self.edges['nodes'][:, 1]] - \
            self.node_coords[self.edges['nodes'][:, 0]]
        vals = _row_dot(e, e) * self.ce_ratios / 6.0

        edge_nodes = self.edges['nodes']
        numpy.add.at(self.control_volumes, edge_nodes[:, 0], vals)
        numpy.add.at(self.control_volumes, edge_nodes[:, 1], vals)

        return

    def _get_face_circumcenter(self, face_id):
        '''Computes the center of the circumcircle of a given face.

        :params face_id: Face ID for which to compute circumcenter.
        :type face_id: int
        :returns circumcenter: Circumcenter of the face with given face ID.
        :type circumcenter: numpy.ndarray((float,3))
        '''
        from vtk import vtkTriangle

        x = self.node_coords[self.faces['nodes'][face_id]]
        # Project triangle to 2D.
        v = numpy.empty(3, dtype=numpy.dtype((float, 2)))
        vtkTriangle.ProjectTo2D(x[0], x[1], x[2],
                                v[0], v[1], v[2])
        # Get the circumcenter in 2D.
        cc_2d = numpy.empty(2, dtype=float)
        vtkTriangle.Circumcircle(v[0], v[1], v[2], cc_2d)
        # Project back to 3D by using barycentric coordinates.
        bcoords = numpy.empty(3, dtype=float)
        vtkTriangle.BarycentricCoords(cc_2d, v[0], v[1], v[2], bcoords)
        return bcoords[0] * x[0] + bcoords[1] * x[1] + bcoords[2] * x[2]

        # a = x[0] - x[1]
        # b = x[1] - x[2]
        # c = x[2] - x[0]
        # w = numpy.cross(a, b)
        # omega = 2.0 * numpy.dot(w, w)
        # if abs(omega) < 1.0e-10:
        #     raise ZeroDivisionError(
        #             'The nodes don''t seem to form a proper triangle.'
        #             )
        # alpha = -numpy.dot(b, b) * numpy.dot(a, c) / omega
        # beta = -numpy.dot(c, c) * numpy.dot(b, a) / omega
        # gamma = -numpy.dot(a, a) * numpy.dot(c, b) / omega
        # m = alpha * x[0] + beta * x[1] + gamma * x[2]

        # # Alternative implementation from
        # # https://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        # a = x[1] - x[0]
        # b = x[2] - x[0]
        # alpha = numpy.dot(a, a)
        # beta = numpy.dot(b, b)
        # w = numpy.cross(a, b)
        # omega = 2.0 * numpy.dot(w, w)
        # m = numpy.empty(3)
        # m[0] = x[0][0] + (
        #         (alpha * b[1] - beta * a[1]) * w[2] -
        #         (alpha * b[2] - beta * a[2]) * w[1]
        #         ) / omega
        # m[1] = x[0][1] + (
        #         (alpha * b[2] - beta * a[2]) * w[0] -
        #         (alpha * b[0] - beta * a[0]) * w[2]
        #         ) / omega
        # m[2] = x[0][2] + (
        #         (alpha * b[0] - beta * a[0]) * w[1] -
        #         (alpha * b[1] - beta * a[1]) * w[0]
        #         ) / omega
        # return

    def num_delaunay_violations(self):
        # is_delaunay = True
        num_faces = len(self.faces['nodes'])
        num_interior_faces = 0
        num_delaunay_violations = 0

        if 'cells' not in self.faces:
            self.create_face_cells()

        for face_id in range(num_faces):
            # Boundary faces don't need to be checked.
            if len(self.faces['cells'][face_id]) != 2:
                continue

            num_interior_faces += 1
            # Each interior edge divides the domain into to half-planes.
            # The Delaunay condition is fulfilled if and only if
            # the circumcenters of the adjacent cells are in "the right order",
            # i.e., line between the nodes of the cells which do not sit
            # on the hyperplane have the same orientation as the line
            # between the circumcenters.

            # The orientation of the coedge needs gauging.
            # Do it in such as a way that the control volume contribution
            # is positive if and only if the area of the triangle
            # (node, other0, edge_midpoint) (in this order) is positive.
            # Equivalently, the triangles (node, edge_midpoint, other1)
            # or (node, other0, other1) could  be considered.
            # other{0,1} refers to the the node opposing the edge in the
            # adjacent cell {0,1}.
            # Get the opposing node of the first adjacent cell.
            cell0 = self.faces['cells'][face_id][0]
            # This nonzero construct is an ugly replacement for the nonexisting
            # index() method. (Compare with Python lists.)
            face_lid = \
                numpy.nonzero(self.cells['faces'][cell0] == face_id)[0][0]
            # This makes use of the fact that cellsEdges and cellsNodes
            # are coordinated such that in cell #i, the edge cellsEdges[i][k]
            # opposes cellsNodes[i][k].
            other0 = self.node_coords[self.cells['nodes'][cell0][face_lid]]

            # Get the edge midpoint.
            node_ids = self.faces['nodes'][face_id]
            node_coords = self.node_coords[node_ids]
            edge_midpoint = 0.5 * (node_coords[0] + node_coords[1])

            # Get the circumcenters of the adjacent cells.
            cc = self.cell_circumcenters[self.faces['cells'][face_id]]
            # Check if cc[1]-cc[0] and the gauge point
            # in the "same" direction.
            if numpy.dot(edge_midpoint-other0, cc[1]-cc[0]) < 0.0:
                num_delaunay_violations += 1
        return num_delaunay_violations

    def show_control_volume(self, node_id):
        '''Displays a node with its surrounding control volume.

        :param node_id: Node ID for which to show the control volume.
        :type node_id: int
        '''
        import os
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        if 'DISPLAY' not in os.environ:
            # headless mode, for remote executions (and travis)
            mpl.use('Agg')
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.axis('equal')

        if 'cells' not in self.faces:
            self.create_face_cells()

        # get cell circumcenters
        cell_ccs = self.cell_circumcenters

        # There are not node->edge relations so manually build the list.
        adjacent_edge_ids = []
        for edge_id, nodes in enumerate(self.edges['nodes']):
            if node_id in nodes:
                adjacent_edge_ids.append(edge_id)

        # Loop over all adjacent edges and plot the edges and their ce_ratios.
        for k, edge_id in enumerate(adjacent_edge_ids):
            # get rainbow color
            h = float(k) / len(adjacent_edge_ids)
            hsv_face_col = numpy.array([[[h, 1.0, 1.0]]])
            col = mpl.colors.hsv_to_rgb(hsv_face_col)[0][0]

            edge_nodes = self.node_coords[self.edges['nodes'][edge_id]]

            # highlight edge
            ax.plot(
                edge_nodes[:, 0], edge_nodes[:, 1], edge_nodes[:, 2],
                color=col, linewidth=3.0
                )

            # edge_midpoint = 0.5 * (edge_nodes[0] + edge_nodes[1])

            # Plot ce_ratio.
            # face_col = '0.7'
            edge_col = 'k'
            for k, face_id in enumerate(self.edges['faces'][edge_id]):
                ccs = cell_ccs[self.faces['cells'][face_id]]
                if len(ccs) == 2:
                    ax.plot(ccs[:, 0], ccs[:, 1], ccs[:, 2], color=edge_col)
                    # tri = mpl3.art3d.Poly3DCollection(
                    #     [numpy.vstack((ccs, edge_midpoint))]
                    #     )
                    # tri.set_color(face_col)
                    # ax.add_collection3d(tri)
                elif len(ccs) == 1:
                    face_cc = self._get_face_circumcenter(face_id)
                    # tri = mpl3.art3d.Poly3DCollection(
                    #     [numpy.vstack((ccs[0], face_cc, edge_midpoint))]
                    #     )
                    # tri.set_color(face_col)
                    # ax.add_collection3d(tri)
                    ax.plot(
                        [ccs[0][0], face_cc[0]],
                        [ccs[0][1], face_cc[1]],
                        [ccs[0][2], face_cc[2]],
                        color=edge_col
                        )
                else:
                    raise RuntimeError('???')
        return

    def show_edge(self, edge_id):
        '''Displays edge with ce_ratio.

        :param edge_id: Edge ID for which to show the ce_ratio.
        :type edge_id: int
        '''
        import os
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        if 'DISPLAY' not in os.environ:
            # headless mode, for remote executions (and travis)
            mpl.use('Agg')
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.axis('equal')

        edge_nodes = self.node_coords[self.edges['nodes'][edge_id]]

        if 'cells' not in self.edges:
            self.create_edge_cells()

        # plot all adjacent cells
        col = 'k'
        for cell_id in self.edges['cells'][edge_id]:
            for edge in self.cells['edges'][cell_id]:
                x = self.node_coords[self.edges['nodes'][edge]]
                ax.plot(x[:, 0], x[:, 1], x[:, 2], col)

        # make clear which is the edge
        ax.plot(edge_nodes[:, 0], edge_nodes[:, 1], edge_nodes[:, 2],
                color=col, linewidth=3.0)

        # get cell circumcenters
        cell_ccs = self.cell_circumcenters

        edge_midpoint = 0.5 * (edge_nodes[0] + edge_nodes[1])

        # plot faces in matching colors
        num_local_faces = len(self.edges['faces'][edge_id])
        for k, face_id in enumerate(self.edges['faces'][edge_id]):
            # get rainbow color
            h = float(k) / num_local_faces
            hsv_face_col = numpy.array([[[h, 1.0, 1.0]]])
            col = mpl.colors.hsv_to_rgb(hsv_face_col)[0][0]

            # paint the face
            import mpl_toolkits.mplot3d as mpl3
            face_nodes = self.node_coords[self.faces['nodes'][face_id]]
            tri = mpl3.art3d.Poly3DCollection([face_nodes])
            tri.set_color(mpl.colors.rgb2hex(col))
            # tri.set_alpha(0.5)
            ax.add_collection3d(tri)

            # mark face circumcenters
            face_cc = self._get_face_circumcenter(face_id)
            ax.plot([face_cc[0]], [face_cc[1]], [face_cc[2]],
                    marker='o', color=col)

        # plot ce_ratio
        face_col = '0.7'
        col = 'k'
        for k, face_id in enumerate(self.edges['faces'][edge_id]):
            ccs = cell_ccs[self.faces['cells'][face_id]]
            if len(ccs) == 2:
                tri = mpl3.art3d.Poly3DCollection([
                    numpy.vstack((ccs, edge_midpoint))
                    ])
                tri.set_color(face_col)
                ax.add_collection3d(tri)
                ax.plot(ccs[:, 0], ccs[:, 1], ccs[:, 2], color=col)
            elif len(ccs) == 1:
                tri = mpl3.art3d.Poly3DCollection(
                    [numpy.vstack((ccs[0], face_cc, edge_midpoint))]
                    )
                tri.set_color(face_col)
                ax.add_collection3d(tri)
                ax.plot([ccs[0][0], face_cc[0]],
                        [ccs[0][1], face_cc[1]],
                        [ccs[0][2], face_cc[2]],
                        color=col)
            else:
                raise RuntimeError('???')

        # ax.plot([edge_midpoint[0]],
        #         [edge_midpoint[1]],
        #         [edge_midpoint[2]],
        #         'ro'
        #         )

        # highlight cells
        highlight_cells = []  # [3]
        col = 'r'
        for k in highlight_cells:
            cell_id = self.edges['cells'][edge_id][k]
            ax.plot([cell_ccs[cell_id, 0]],
                    [cell_ccs[cell_id, 1]],
                    [cell_ccs[cell_id, 2]],
                    color=col,
                    marker='o'
                    )
            for edge in self.cells['edges'][cell_id]:
                x = self.node_coords[self.edges['nodes'][edge]]
                ax.plot(x[:, 0], x[:, 1], x[:, 2], col, linestyle='dashed')
        return
