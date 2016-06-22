# -*- coding: utf-8 -*-
import numpy as np

import mesh2d


class mesh2D_shell(mesh2d):
    def __init__(self,
                 nodes,
                 cellsNodes,
                 cellsEdges=None,
                 edgesNodes=None,
                 edgesCells=None
                 ):
        # It would be sweet if we could handle cells and the rest as arrays
        # with fancy dtypes such as
        #
        #     np.dtype([('nodes', (int, num_local_nodes)),
        #               ('edges', (int, num_local_edges))]),
        #
        # but right now there's no (no easy?) way to extend nodes properly
        # for the case that 'edges' aren't given. A simple recreate-and-copy
        #
        #     for k in xrange(num_cells):
        #         new_cells[k]['nodes'] = self.cells[k]['nodes']
        #
        # does not seem to work for whatever reason.
        # Hence, handle cells and friends of dictionaries of np.arrays.
        if not isinstance(nodes, np.ndarray):
            raise TypeError(
                    'For performace reasons, '
                    'build nodes as '
                    'np.empty(num_nodes, dtype=np.dtype((float, 3)))'
                    )

        if not isinstance(cellsNodes, np.ndarray):
            raise TypeError(
                'For performace reasons, '
                'build cellsNodes as '
                'np.empty(num_nodes, dtype=np.dtype((int, 3)))'
                )

        if cellsEdges is not None and not isinstance(cellsEdges, np.ndarray):
            raise TypeError(
                'For performace reasons, '
                'build cellsEdges as '
                'np.empty(num_nodes, dtype=np.dtype((int, 3)))'
                )

        if edgesNodes is not None and not isinstance(edgesNodes, np.ndarray):
            raise TypeError(
                'For performace reasons, '
                'build edgesNodes as '
                'np.empty(num_nodes, dtype=np.dtype((int, 2)))'
                )

        self.nodes = nodes
        self.edgesNodes = edgesNodes
        self.edgesCells = edgesCells
        self.cellsNodes = cellsNodes
        self.cellsEdges = cellsEdges

        self.cellsVolume = None
        self.cell_circumcenters = None
        self.control_volumes = None

        self.vtk_mesh = None

    def create_cells_volume(self):
        '''Returns the area of triangle spanned by the two given edges.'''
        import vtk
        num_cells = len(self.cellsNodes)
        self.cellsVolume = np.empty(num_cells, dtype=float)
        for cell_id, cellNodes in enumerate(self.cellsNodes):
            # edge0 = node0 - node1
            # edge1 = node1 - node2
            # self.cellsVolume[cell_id] = \
            #    0.5 * np.linalg.norm( np.cross( edge0, edge1 ) )
            x = self.nodes[cellNodes]
            self.cellsVolume[cell_id] = \
                abs(vtk.vtkTriangle.TriangleArea(x[0], x[1], x[2]))
        return

    def create_cell_circumcenters(self):
        '''Computes the center of the circumsphere of each cell.
        '''
        import vtk
        num_cells = len(self.cellsNodes)
        self.cell_circumcenters = \
            np.empty(num_cells, dtype=np.dtype((float, 3)))
        for cell_id, cellNodes in enumerate(self.cellsNodes):
            x = self.nodes[cellNodes]
            # Project triangle to 2D.
            v = np.empty(3, dtype=np.dtype((float, 2)))
            vtk.vtkTriangle.ProjectTo2D(x[0], x[1], x[2],
                                        v[0], v[1], v[2])
            # Get the circumcenter in 2D.
            cc_2d = np.empty(2, dtype=float)
            vtk.vtkTriangle.Circumcircle(v[0], v[1], v[2],
                                         cc_2d)
            # Project back to 3D by using barycentric coordinates.
            bcoords = np.empty(3, dtype=float)
            vtk.vtkTriangle.BarycentricCoords(cc_2d, v[0], v[1], v[2], bcoords)
            self.cell_circumcenters[cell_id] = \
                bcoords[0] * x[0] + bcoords[1] * x[1] + bcoords[2] * x[2]

            # a = x[0] - x[1]
            # b = x[1] - x[2]
            # c = x[2] - x[0]
            # w = np.cross(a, b)
            # omega = 2.0 * np.dot(w, w)
            # if abs(omega) < 1.0e-10:
            #     raise ZeroDivisionError('The nodes don''t seem to form '
            #                             'a proper triangle.')
            # alpha = -np.dot(b, b) * np.dot(a, c) / omega
            # beta  = -np.dot(c, c) * np.dot(b, a) / omega
            # gamma = -np.dot(a, a) * np.dot(c, b) / omega
            # m = alpha * x[0] + beta * x[1] + gamma * x[2]

            # # Alternative implementation from
            # # https://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
            # a = x[1] - x[0]
            # b = x[2] - x[0]
            # alpha = np.dot(a, a)
            # beta = np.dot(b, b)
            # w = np.cross(a, b)
            # omega = 2.0 * np.dot(w, w)
            # m = np.empty(3)
            # m[0] = x[0][0] + ((alpha * b[1] - beta * a[1]) * w[2]
            #                   -(alpha * b[2] - beta * a[2]) * w[1]) / omega
            # m[1] = x[0][1] + ((alpha * b[2] - beta * a[2]) * w[0]
            #                 -(alpha * b[0] - beta * a[0]) * w[2]) / omega
            # m[2] = x[0][2] + ((alpha * b[0] - beta * a[0]) * w[1]
            #                 -(alpha * b[1] - beta * a[1]) * w[0]) / omega

        return

    def compute_control_volumes(self):
        num_nodes = len(self.nodes)
        self.control_volumes = np.zeros((num_nodes, 1), dtype=float)

        # compute cell circumcenters
        if self.cell_circumcenters is None:
            self.create_cell_circumcenters()
        circumcenters = self.cell_circumcenters

        if self.edgesNodes is None:
            self.create_adjacent_entities()

        # Precompute edge lengths.
        num_edges = len(self.edgesNodes)
        edge_lengths = np.empty(num_edges, dtype=float)
        for k in xrange(num_edges):
            nodes = self.nodes[self.edgesNodes[k]]
            edge_lengths[k] = np.linalg.norm(nodes[1] - nodes[0])

        # Get edge normals.
        edge_normals = self._compute_edge_normals(edge_lengths)

        # Compute covolumes and control volumes.
        for k in xrange(num_edges):
            # Get the circumcenters of the adjacent cells.
            cc = circumcenters[self.edgesCells[k]]
            node_ids = self.edgesNodes[k]
            if len(cc) == 2:  # interior cell
                # TODO check out if this holds true for bent surfaces too
                coedge = cc[1] - cc[0]
            elif len(cc) == 1:  # boundary cell
                node_coords = self.nodes[node_ids]
                edge_midpoint = 0.5 * (node_coords[0] + node_coords[1])
                coedge = edge_midpoint - cc[0]
            else:
                raise RuntimeError(
                    'An edge should have either 1 or two adjacent cells.'
                    )

            # Project the coedge onto the outer normal. The two vectors should
            # be parallel, it's just the sign of the coedge length that is to
            # be determined here.
            covolume = np.dot(coedge, edge_normals[k])
            pyramid_volume = 0.5 * edge_lengths[k] * covolume / 2
            self.control_volumes[node_ids] += pyramid_volume

        return

    def _compute_edge_normals(self, edge_lengths):
        # Precompute edge normals. Do that in such a way that the
        # face normals points in the direction of the cell with the higher
        # cell ID.
        num_edges = len(self.edgesNodes)
        edge_normals = np.empty(num_edges, dtype=np.dtype((float, 3)))
        for cell_id, cellEdges in enumerate(self.cellsEdges):
            # Loop over the local faces.
            for k in xrange(3):
                edge_id = cellEdges[k]
                # Compute the normal in the direction of the higher cell ID,
                # or if this is a boundary face, to the outside of the domain.
                neighbor_cell_ids = self.edgesCells[edge_id]
                if cell_id == neighbor_cell_ids[0]:
                    edge_nodes = self.nodes[self.edgesNodes[edge_id]]
                    # The current cell is the one with the lower ID.
                    # Get "other" node (aka the one which is not in the current
                    # "face").
                    other_node_id = self.cellsNodes[cell_id][k]
                    # Get any direction other_node -> face.
                    # As reference, any point in face can be taken, e.g.,
                    # the first face corner point
                    # self.edgesNodes[edge_id][0].
                    edge_normals[edge_id] = \
                        edge_nodes[0] - self.nodes[other_node_id]
                    # Make it orthogonal to the face.
                    edge_dir = (edge_nodes[1] - edge_nodes[0]) / \
                        edge_lengths[edge_id]
                    edge_normals[edge_id] -= np.dot(
                            edge_normals[edge_id], edge_dir
                            ) * edge_dir
                    # Normalization.
                    edge_normals[edge_id] /= \
                        np.linalg.norm(edge_normals[edge_id])
        return edge_normals

    def show(self, highlight_nodes=None):
        '''Plot the mesh.'''
        if highlight_nodes is None:
            highlight_nodes = []

        if self.edgesNodes is None:
            raise RuntimeError('Can only show mesh when edges are created.')

        import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D

        if self.cell_circumcenters is None:
            self.create_cell_circumcenters()

        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.gca()
        plt.axis('equal')

        # plot edges
        col = 'k'
        for node_ids in self.edgesNodes:
            x = self.nodes[node_ids]
            ax.plot(x[:, 0],
                    x[:, 1],
                    # x[:,2],
                    col)

        # Highlight nodes and their covolumes.
        node_col = 'r'
        covolume_col = '0.5'
        for node_id in highlight_nodes:
            x = self.nodes[node_id]
            ax.plot([x[0]],
                    [x[1]],
                    [x[2]],
                    color=node_col, marker='o')
            # Plot the covolume.
            # TODO Something like nodesEdges would be useful here.
            for edge_id, node_ids in enumerate(self.edgesNodes):
                if node_id in node_ids:
                    adjacent_cells = self.edgesCells[edge_id]
                    ccs = self.cell_circumcenters[adjacent_cells]
                    if len(ccs) == 2:
                        ax.plot([ccs[0][0], ccs[1][0]],
                                [ccs[0][1], ccs[1][1]],
                                [ccs[0][2], ccs[1][2]],
                                color=covolume_col
                                )
                    elif len(ccs) == 1:
                        edge_midpoint = 0.5 * (
                            self.nodes[node_ids[0]] + self.nodes[node_ids[1]]
                            )
                        ax.plot([ccs[0][0], edge_midpoint[0]],
                                [ccs[0][1], edge_midpoint[1]],
                                [ccs[0][2], edge_midpoint[2]],
                                color=covolume_col
                                )
                    else:
                        raise RuntimeError(
                            'An edge has to have either 1 or 2 adjacent cells.'
                            )

        plt.show()
        return
