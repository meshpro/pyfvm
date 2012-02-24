# -*- coding: utf-8 -*-
# ==============================================================================
import numpy as np
from mesh import Mesh
# ==============================================================================
class Mesh2D( Mesh ):
    # --------------------------------------------------------------------------
    def __init__(self,
                 nodes,
                 cellsNodes,
                 cellsEdges = None,
                 edgesNodes = None,
                 edgesCells = None
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
        if not isinstance(nodes,np.ndarray):
           raise TypeError('For performace reasons, build nodes as np.empty(num_nodes, dtype=np.dtype((float, 3)))')

        if not isinstance(cellsNodes,np.ndarray):
           raise TypeError('For performace reasons, build cellsNodes as np.empty(num_nodes, dtype=np.dtype((int, 3)))')

        if cellsEdges is not None and not isinstance(cellsEdges,np.ndarray):
           raise TypeError('For performace reasons, build cellsEdges as np.empty(num_nodes, dtype=np.dtype((int, 3)))')

        if edgesNodes is not None and  not isinstance(edgesNodes,np.ndarray):
           raise TypeError('For performace reasons, build edgesNodes as np.empty(num_nodes, dtype=np.dtype((int, 2)))')

        self.nodes = nodes
        self.edgesNodes = edgesNodes
        self.edgesCells = edgesCells
        self.cellsNodes = cellsNodes
        self.cellsEdges = cellsEdges

        self.cellsVolume = None
        self.cell_circumcenters = None
        self.control_volumes = None

        self.vtk_mesh = None
    # --------------------------------------------------------------------------
    def create_cells_volume(self):
        '''Returns the area of triangle spanned by the two given edges.'''
        import vtk
        num_cells = len(self.cellsNodes)
        self.cellsVolume = np.empty(num_cells, dtype=float)
        for cell_id, cellNodes in enumerate(self.cellsNodes):
            #edge0 = node0 - node1
            #edge1 = node1 - node2
            #self.cellsVolume[cell_id] = 0.5 * np.linalg.norm( np.cross( edge0, edge1 ) )
            # Append a third component.
            z = np.zeros((3,1))
            x = np.c_[self.nodes[cellNodes], z]
            self.cellsVolume[cell_id] = \
               abs(vtk.vtkTriangle.TriangleArea(x[0], x[1], x[2]))
        return
    # --------------------------------------------------------------------------
    def create_cell_circumcenters( self ):
        '''Computes the center of the circumsphere of each cell.
        '''
        import vtk
        num_cells = len(self.cellsNodes)
        self.cell_circumcenters = np.empty(num_cells, dtype=np.dtype((float,2)))
        for cell_id, cellNodes in enumerate(self.cellsNodes):
            x = self.nodes[cellNodes]
            vtk.vtkTriangle.Circumcircle(x[0], x[1], x[2],
                                         self.cell_circumcenters[cell_id])

        return
    # --------------------------------------------------------------------------
    def create_adjacent_entities( self ):

        if self.edgesNodes is not None:
            return

        # Take cell #0 as representative.
        num_local_nodes = len(self.cellsNodes[0])
        num_local_edges = num_local_nodes * (num_local_nodes-1) / 2
        num_cells = len(self.cellsNodes)

        # Get upper bound for number of edges; trim later.
        max_num_edges = num_local_nodes * num_cells
        self.edgesNodes = np.empty(max_num_edges, dtype=np.dtype((int,2)))
        self.edgesCells = [[] for k in xrange(max_num_edges)]

        self.cellsEdges = np.empty(num_cells, dtype=np.dtype((int,num_local_edges)))

        # The (sorted) dictionary edges keeps track of how nodes and edges
        # are connected.
        # If  node_edges[(3,4)] == 17  is true, then the nodes (3,4) are
        # connected  by edge 17.
        edges = {}

        new_edge_gid = 0
        # Loop over all elements.
        for cell_id in xrange(num_cells):
            # We're treating simplices so loop over all combinations of
            # local nodes.
            # Make sure cellNodes are sorted.
            self.cellsNodes[cell_id] = np.sort(self.cellsNodes[cell_id])
            for k in xrange(len(self.cellsNodes[cell_id])):
                # Remove the k-th element. This makes sure that the k-th
                # edge is opposite of the k-th node. Useful later in
                # in construction of edge (face) normals.
                indices = tuple(self.cellsNodes[cell_id][:k]) \
                        + tuple(self.cellsNodes[cell_id][k+1:])
                if indices in edges:
                    edge_gid = edges[indices]
                    self.edgesCells[edge_gid].append( cell_id )
                    self.cellsEdges[cell_id][k] = edge_gid
                else:
                    # add edge
                    self.edgesNodes[new_edge_gid] = indices
                    # edgesCells is also always ordered.
                    self.edgesCells[new_edge_gid].append( cell_id )
                    self.cellsEdges[cell_id][k] = new_edge_gid
                    edges[indices] = new_edge_gid
                    new_edge_gid += 1

        # trim edges
        self.edgesNodes = self.edgesNodes[:new_edge_gid]
        self.edgesCells = self.edgesCells[:new_edge_gid]

        return
    # --------------------------------------------------------------------------
    def refine( self ):
        '''Canonically refine a mesh by inserting nodes at all edge midpoints
        and make four triangular elements where there was one.'''
        if self.edgesNodes is None:
            raise RuntimeError("Edges must be defined to do refinement.")

        # Record the newly added nodes.
        num_new_nodes = len(self.edgesNodes)
        new_nodes = np.empty(num_new_nodes, dtype=np.dtype((float,3)))
        new_node_gid = len(self.nodes)

        # After the refinement step, all previous edge-node associations will
        # be obsolete, so record *all* the new edges.
        num_new_edges = 2 * len(self.edgesNodes) + 3 * len(self.cellsNodes)
        new_edgesNodes = np.empty(num_new_edges, dtype=np.dtype((int,2)))
        new_edge_gid = 0

        # After the refinement step, all previous cell-node associations will
        # be obsolete, so record *all* the new cells.
        num_new_cells = 4 * len(self.cellsNodes)
        new_cellsNodes = np.empty(num_new_cells, dtype=np.dtype((int,3)))
        new_cellsEdges = np.empty(num_new_cells, dtype=np.dtype((int,3)))
        new_cell_gid = 0

        num_edges = len(self.edgesNodes)
        is_edge_divided = np.zeros(num_edges, dtype=bool)
        edge_midpoint_gids = np.empty(num_edges, dtype=int)
        dt = np.dtype((int,2))
        edge_newedges_gids = np.empty(num_edges, dtype=dt)
        # Loop over all elements.
        if self.cellsEdges is None or len(self.cellsEdges) != len(self.cellsNodes):
            raise RuntimeError("Edges must be defined for each cell.")
        for cellNodes, cellEdges in zip(self.cellsNodes,self.cellsEdges):
            # Divide edges.
            num_local_edges = len(cellEdges)
            local_edge_midpoint_gids = np.empty(num_local_edges, dtype=int)
            local_edge_newedges = np.empty(num_local_edges, dtype=dt)
            local_neighbor_midpoints = [ [], [], [] ]
            local_neighbor_newedges = [ [], [], [] ]
            for k, local_edge_gid in enumerate(cellEdges):
                edgenodes_gids = self.edgesNodes[local_edge_gid]
                if is_edge_divided[local_edge_gid]:
                    # Edge is already divided. Just keep records
                    # for the cell creation.
                    local_edge_midpoint_gids[k] = \
                        edge_midpoint_gids[local_edge_gid]
                    local_edge_newedges[k] = edge_newedges[local_edge_gid]
                else:
                    # Create new node at the edge midpoint.
                    print new_nodes[new_node_gid]
                    new_nodes[new_node_gid] = \
                        0.5 * (self.nodes[edgenodes_gids[0]] \
                              +self.nodes[edgenodes_gids[1]])
                    local_edge_midpoint_gids[k] = new_node_gid
                    new_node_gid += 1
                    edge_midpoint_gids[local_edge_gid] = \
                        local_edge_midpoint_gids[k]

                    # Divide edge into two.
                    new_edgesNodes[new_edge_gid] = \
                        np.array([edgenodes_gids[0], local_edge_midpoint_gids[k]])
                    new_edge_gid += 1
                    new_edgesNodes[new_edge_gid] = \
                        np.array([local_edge_midpoint_gids[k], edgenodes_gids[1]])
                    new_edge_gid += 1

                    local_edge_newedges[k] = \
                        np.array([new_edge_gid-2, new_edge_gid-1])
                    edge_newedges_gids[local_edge_gid] = \
                        local_edge_newedges[k]
                    # Do the household.
                    is_edge_divided[local_edge_gid] = True
                # Keep a record of the new neighbors of the old nodes.
                # Get local node IDs.
                edgenodes_lids = [cellNodes.index(edgenodes_gids[0]),
                                  cellNodes.index(edgenodes_gids[1])]
                local_neighbor_midpoints[edgenodes_lids[0]] \
                    .append( local_edge_midpoint_gids[k] )
                local_neighbor_midpoints[edgenodes_lids[1]]\
                    .append( local_edge_midpoint_gids[k] )
                local_neighbor_newedges[edgenodes_lids[0]] \
                    .append( local_edge_newedges[k][0] )
                local_neighbor_newedges[edgenodes_lids[1]] \
                    .append( local_edge_newedges[k][1] )

            new_edge_opposite_of_local_node = np.empty(3, dtype=int)
            # New edges: Connect the three midpoints.
            for k in xrange(3):
                new_edgesNodes[new_edge_gid] = local_neighbor_midpoints[k]
                new_edge_opposite_of_local_node[k] = new_edge_gid
                new_edge_gid += 1

            # Create new elements.
            # Center cell:
            new_cellsNodes[new_cell_gid] = local_edge_midpoint_gids
            new_cellsEdges[new_cell_gid] = new_edge_opposite_of_local_node
            new_cell_gid += 1
            # The three corner elements:
            for k in xrange(3):
                new_cellsNodes[new_cell_gid] = \
                    np.array([cell.node_indices[k],
                              local_neighbor_midpoints[k][0],
                              local_neighbor_midpoints[k][1]])
                new_cellsEdges[new_cell_gid] = \
                    np.array([new_edge_opposite_of_local_node[k],
                              local_neighbor_newedges[k][0],
                              local_neighbor_newedges[k][1]])
                new_cell_gid += 1

        np.append(self.nodes, new_nodes)
        self.edgesNodes = new_edgesNodes
        self.cellsNodes = new_cellsNodes
        self.cellsEdges = new_cellsEdges
        return
    # --------------------------------------------------------------------------
    def compute_control_volumes( self ):
        num_nodes = len(self.nodes)
        self.control_volumes = np.zeros((num_nodes,1), dtype = float)

        # compute cell circumcenters
        if self.cell_circumcenters is None:
            self.create_cell_circumcenters()

        if self.edgesNodes is None:
            self.create_adjacent_entities()

        # Compute covolumes and control volumes.
        num_edges = len(self.edgesNodes)
        for edge_id in xrange(num_edges):
            # Move the system such that one of the two end points is in the
            # origin. Deliberately take self.edgesNodes[edge_id][0].
            node = self.nodes[self.edgesNodes[edge_id][0]]

            # The orientation of the coedge needs gauging.
            # Do it in such as a way that the control volume contribution
            # is positive if and only if the area of the triangle
            # (node, other0, edge_midpoint) (in this order) is positive.
            # Equivalently, the triangle (node, edge_midpoint, other1) could
            # be considered.
            # other{0,1} refer to that one node of the adjacent.
            # Get the opposing node of the first adjacent cell.
            cell0 = self.edgesCells[edge_id][0]
            edge_idx = np.nonzero(self.cellsEdges[cell0] == edge_id)[0][0]
            # This makes use of the fact that cellsEdges and cellsNodes
            # are coordinated such that in cell #i, the edge cellsEdges[i][k]
            # opposes cellsNodes[i][k].
            other0 = self.nodes[self.cellsNodes[cell0][edge_idx]] \
                   - node
            node_ids = self.edgesNodes[edge_id]
            node_coords = self.nodes[node_ids]
            edge_midpoint = 0.5 * (node_coords[0] + node_coords[1]) \
                          - node
            # Computing the triangle volume like this is called the shoelace
            # formula and can be interpreted as the z-component of the
            # cross-product of other0 and edge_midpoint.
            gauge = other0[0] * edge_midpoint[1] \
                  - other0[1] * edge_midpoint[0]

            # Get the circumcenters of the adjacent cells.
            cc = self.cell_circumcenters[self.edgesCells[edge_id]] \
               - node
            if len(cc) == 2: # interior edge
                self.control_volumes[node_ids] += np.sign(gauge) \
                                                * 0.5 * (cc[0][0] * cc[1][1] \
                                                        -cc[0][1] * cc[1][0])
            elif len(cc) == 1: # boundary edge
                self.control_volumes[node_ids] += np.sign(gauge) \
                                                * 0.5 * (cc[0][0] * edge_midpoint[1]
                                                        -cc[0][1] * edge_midpoint[0])
            else:
                raise RuntimeError('An edge should have either 1 or two adjacent cells.')

        return
    # --------------------------------------------------------------------------
    def _compute_edge_normals(self, edge_lengths):
        # Precompute edge normals. Do that in such a way that the
        # face normals points in the direction of the cell with the higher
        # cell ID.
        num_edges = len(self.edgesNodes)
        edge_normals = np.empty(num_edges, dtype=np.dtype((float,2)))
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
                    edge_normals[edge_id] = edge_nodes[0] \
                                      - self.nodes[other_node_id]
                    # Make it orthogonal to the face.
                    edge_dir = (edge_nodes[1] - edge_nodes[0]) / edge_lengths[edge_id]
                    edge_normals[edge_id] -= np.dot(edge_normals[edge_id], edge_dir) * edge_dir
                    # Normalization.
                    edge_normals[edge_id] /= np.linalg.norm(edge_normals[edge_id])

        return edge_normals
    # --------------------------------------------------------------------------
    def show(self, highlight_nodes = []):
        '''Plot the mesh.'''
        if self.edgesNodes is None:
            raise RuntimeError('Can only show mesh when edges are created.')

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if self.cell_circumcenters is None:
            self.create_cell_circumcenters()

        fig = plt.figure()
        #ax = fig.gca(projection='3d')
        ax = fig.gca()
        plt.axis('equal')

        # plot edges
        col = 'k'
        for node_ids in self.edgesNodes:
            x = self.nodes[node_ids]
            ax.plot(x[:,0],
                    x[:,1],
                    #x[:,2],
                    col)

        # Highlight nodes and their covolumes.
        node_col = 'r'
        covolume_col = '0.5'
        for node_id in highlight_nodes:
            x = self.nodes[node_id]
            ax.plot([x[0]],
                    [x[1]],
                    color=node_col, marker='o')
            # Plot the covolume.
            # TODO Something like nodesEdges would be useful here.
            for edge_id, node_ids in enumerate(self.edgesNodes):
                if node_id in node_ids:
                    adjacent_cells = self.edgesCells[edge_id]
                    ccs = self.cell_circumcenters[adjacent_cells]
                    if len(ccs) == 2:
                        ax.plot( [ccs[0][0], ccs[1][0]],
                                 [ccs[0][1], ccs[1][1]],
                                 color = covolume_col )
                    elif len(ccs) == 1:
                        edge_midpoint = 0.5 * (self.nodes[node_ids[0]]
                                              +self.nodes[node_ids[1]])
                        ax.plot( [ccs[0][0], edge_midpoint[0]],
                                 [ccs[0][1], edge_midpoint[1]],
                                 color = covolume_col )
                    else:
                        raise RuntimeError('An edge has to have either 1 or 2 adjacent cells.')

        plt.show()
        return
    # --------------------------------------------------------------------------
# ==============================================================================
