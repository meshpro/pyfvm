# -*- coding: utf-8 -*-
#
import numpy
from pyfvm.base import _base_mesh, _row_dot

__all__ = ['meshTri']


class meshTri(_base_mesh):
    '''Class for handling triangular meshes.

    .. inheritance-diagram:: meshTri
    '''
    def __init__(self, nodes, cells):
        '''Initialization.
        '''
        # Make sure to only to include those vertices which are part of a cell
        uvertices, uidx = numpy.unique(cells, return_inverse=True)
        cells = uidx.reshape(cells.shape)
        nodes = nodes[uvertices]

        super(meshTri, self).__init__(nodes, cells)
        self.cells = numpy.empty(
                len(cells),
                dtype=numpy.dtype([('nodes', (int, 3))])
                )
        self.cells['nodes'] = cells

        self.create_edges()
        self.compute_edge_lengths()
        self.compute_cell_volumes_and_ce_ratios()
        self.compute_control_volumes()

        self.mark_default_subdomains()

        self.compute_surface_areas()

        self.cell_circumcenters = None

        return

    def mark_default_subdomains(self):
        self.subdomains = {}
        self.subdomains['everywhere'] = {
                'vertices': range(len(self.node_coords)),
                'edges': range(len(self.edges['nodes']))
                }

        # Get vertices on the boundary edges
        boundary_edges = numpy.where(self.is_boundary_edge)[0]
        boundary_vertices = numpy.unique(
                self.edges['nodes'][boundary_edges].flatten()
                )

        self.subdomains['boundary'] = {
                'vertices': boundary_vertices,
                'edges': boundary_edges
                }

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
        _, idx, inv, cts = numpy.unique(
                b,
                return_index=True,
                return_inverse=True,
                return_counts=True
                )
        edge_nodes = a[idx]

        self.is_boundary_edge = (cts == 1)

        self.edges = {
            'nodes': edge_nodes,
            }

        # cell->edges relationship
        num_cells = len(self.cells['nodes'])
        cells_edges = inv.reshape([3, num_cells]).T
        cells = self.cells['nodes']
        self.cells = {
            'nodes': cells,
            'edges': cells_edges
            }

        # store inv for possible later use in create_edge_cells
        self._inv = inv

        return

    def create_edge_cells(self):
        # Create edge->cells relationships
        num_cells = len(self.cells['nodes'])
        edge_cells = [[] for k in range(len(self.edges['nodes']))]
        for k, edge_id in enumerate(self._inv):
            edge_cells[edge_id].append(k % num_cells)
        self.edges['cells'] = edge_cells
        return

    def compute_control_volumes(self):
        '''Compute the control volumes of all nodes in the mesh.
        '''
        e = self.node_coords[self.edges['nodes'][:, 1]] - \
            self.node_coords[self.edges['nodes'][:, 0]]

        #   0.5 * (0.5 * edge_length) * covolume
        # = 0.5 * (0.5 * edge_length**2) * ce_ratio_edge_ratio
        vals = 0.25 * _row_dot(e, e) * self.ce_ratios

        edge_nodes = self.edges['nodes']

        self.control_volumes = numpy.zeros(len(self.node_coords), dtype=float)
        numpy.add.at(self.control_volumes, edge_nodes[:, 0], vals)
        numpy.add.at(self.control_volumes, edge_nodes[:, 1], vals)

        return

    def compute_cell_volumes_and_ce_ratios(self):
        edges = \
            self.node_coords[self.edges['nodes'][:, 1]] - \
            self.node_coords[self.edges['nodes'][:, 0]]
        cells_edges = edges[self.cells['edges']]
        e0 = cells_edges[:, 0, :]
        e1 = cells_edges[:, 1, :]
        e2 = cells_edges[:, 2, :]
        self.cell_volumes, sol = \
            self.compute_tri_areas_and_ce_ratios(e0, e1, e2)

        num_edges = len(self.edges['nodes'])
        self.ce_ratios = numpy.zeros(num_edges, dtype=float)
        numpy.add.at(
                self.ce_ratios,
                self.cells['edges'],
                sol
                )

        return

    def compute_surface_areas(self):
        self.surface_areas = numpy.zeros(len(self.get_vertices('everywhere')))
        b_edge = self.get_edges('boundary')
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

    def compute_gradient(self, u):
        '''Computes an approximation to the gradient :math:`\\nabla u` of a
        given scalar valued function :math:`u`, defined in the node points.
        This is taken from :cite:`NME2187`,

           Discrete gradient method in solid mechanics,
           Lu, Jia and Qian, Jing and Han, Weimin,
           International Journal for Numerical Methods in Engineering,
           http://dx.doi.org/10.1002/nme.2187.
        '''
        if self.cell_circumcenters is None:
            X = self.node_coords[self.cells['nodes']]
            self.cell_circumcenters = self.compute_triangle_circumcenters(X)

        if 'cells' not in self.edges:
            self.create_edge_cells()

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
        for node in self.get_vertices('boundary'):
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
                coedge_midpoint = 0.5 * (
                        cell_circumcenters2d[cell0] +
                        edge_midpoint
                        )
            elif len(self.edges['cells'][edge_id]) == 2:
                cell0 = self.edges['cells'][edge_id][0]
                cell1 = self.edges['cells'][edge_id][1]
                # Interior edge.
                coedge_midpoint = 0.5 * (
                        cell_circumcenters2d[cell0] +
                        cell_circumcenters2d[cell1]
                        )
            else:
                raise RuntimeError(
                        'Edge needs to have either one or two neighbors.'
                        )

            # Compute the coefficient r for both contributions
            coeffs = self.ce_ratios[edge_id] / \
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

    def num_delaunay_violations(self):
        # Delaunay violations are present exactly on the interior edges where
        # the ce_ratio is negative. Count those.
        return numpy.sum(self.ce_ratios[~self.is_boundary_edge] < 0.0)

    def show(self, show_ce_ratios=True):
        '''Show the mesh using matplotlib.

        :param show_ce_ratios: If true, show all ce_ratios of the mesh, too.
        :type show_ce_ratios: bool, optional
        '''
        # Importing matplotlib takes a while, so don't do that at the header.
        import os
        if 'DISPLAY' not in os.environ:
            import matplotlib as mpl
            # headless mode, for remote executions (and travis)
            mpl.use('Agg')
        from matplotlib import pyplot as plt

        # from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.gca()
        plt.axis('equal')

        # plot edges
        for node_ids in self.edges['nodes']:
            x = self.node_coords[node_ids]
            ax.plot(x[:, 0], x[:, 1], 'k')

        if show_ce_ratios:
            # Connect all cell circumcenters with the edge midpoints
            if self.cell_circumcenters is None:
                X = self.node_coords[self.cells['nodes']]
                self.cell_circumcenters = \
                    self.compute_triangle_circumcenters(X)
            edge_midpoints = 0.5 * (
                self.node_coords[self.edges['nodes'][:, 0]] +
                self.node_coords[self.edges['nodes'][:, 1]]
                )
            for cell_id, edges in enumerate(self.cells['edges']):
                for edge_id in edges:
                    p = numpy.c_[
                            self.cell_circumcenters[cell_id],
                            edge_midpoints[edge_id],
                            ]
                    ax.plot(p[0], p[1], color='0.8')
        return

    def show_vertex(self, node_id, show_ce_ratio=True):
        '''Plot the vicinity of a node and its ce_ratio.

        :param node_id: Node ID of the node to be shown.
        :type node_id: int

        :param show_ce_ratio: If true, shows the ce_ratio of the node, too.
        :type show_ce_ratio: bool, optional
        '''
        # Importing matplotlib takes a while, so don't do that at the header.
        import os
        if 'DISPLAY' not in os.environ:
            import matplotlib as mpl
            # headless mode, for remote executions (and travis)
            mpl.use('Agg')
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.gca()
        plt.axis('equal')

        # Find the edges that contain the vertex
        edge_ids = numpy.where((self.edges['nodes'] == node_id).any(axis=1))[0]
        # ... and plot them
        for node_ids in self.edges['nodes'][edge_ids]:
            x = self.node_coords[node_ids]
            ax.plot(x[:, 0], x[:, 1], 'k')

        # Highlight ce_ratios.
        if show_ce_ratio:
            if self.cell_circumcenters is None:
                X = self.node_coords[self.cells['nodes']]
                self.cell_circumcenters = \
                    self.compute_triangle_circumcenters(X)

            # Find the cells that contain the vertex
            cell_ids = numpy.where(
                (self.cells['nodes'] == node_id).any(axis=1)
                )[0]

            for cell_id in cell_ids:
                for edge_id in self.cells['edges'][cell_id]:
                    if node_id not in self.edges['nodes'][edge_id]:
                        continue
                    node_ids = self.edges['nodes'][edge_id]
                    edge_midpoint = 0.5 * (
                            self.node_coords[node_ids[0]] +
                            self.node_coords[node_ids[1]]
                            )
                    p = numpy.c_[
                        self.cell_circumcenters[cell_id],
                        edge_midpoint
                        ]
                    q = numpy.c_[
                        self.cell_circumcenters[cell_id],
                        edge_midpoint,
                        self.node_coords[node_id]
                        ]
                    ax.fill(q[0], q[1], color='0.5')
                    ax.plot(p[0], p[1], color='0.7')
        return
