# -*- coding: utf-8 -*-
# ==============================================================================
__all__=['mesh3d']

import numpy as np
from base import _base_mesh
# ==============================================================================
class mesh3d(_base_mesh):
    # --------------------------------------------------------------------------
    def __init__( self,
                  nodes,
                  cellsNodes,
                  cellsEdges = None,
                  edgesNodes = None,
                  edgesCells = None):
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
        self.edgesFaces = None
        self.edgesCells = edgesCells
        self.facesNodes = None
        self.facesEdges = None
        self.facesCells = None
        self.cellsNodes = cellsNodes
        self.cellsEdges = cellsEdges
        self.cellsFaces = None
        self.cell_circumcenters = None
        self.cellsVolume = None
        self.control_volumes = None
    # --------------------------------------------------------------------------
    def create_cells_volume(self):
        '''Returns the volume of a tetrahedron given by the nodes.
        '''
        import vtk
        num_cells = len(self.cellsNodes)
        self.cellsVolume = np.empty(num_cells, dtype=float)
        for cell_id, cellNodes in enumerate(self.cellsNodes):
            #edge0 = node0 - node1
            #edge1 = node1 - node2
            #edge2 = node2 - node3
            #edge3 = node3 - node0

            #alpha = np.vdot( edge0, np.cross(edge1, edge2) )
            #norm_prod = np.linalg.norm(edge0) \
                      #* np.linalg.norm(edge1) \
                      #* np.linalg.norm(edge2)
            #if abs(alpha) / norm_prod < 1.0e-5:
                ## Edges probably conplanar. Take a different set.
                #alpha = np.vdot( edge0, np.cross(edge1, edge3) )
                #norm_prod = np.linalg.norm(edge0) \
                          #* np.linalg.norm(edge1) \
                          #* np.linalg.norm(edge3)

            #self.cellsVolume[cell_id] = abs( alpha ) / 6.0

            x = self.nodes[cellNodes]
            self.cellsVolume[cell_id] = \
                abs(vtk.vtkTetra.ComputeVolume(x[0], x[1], x[2], x[3]))
        return
    # --------------------------------------------------------------------------
    def create_adjacent_entities( self ):
        # Take cell #0 as representative.
        num_local_nodes = len(self.cellsNodes[0])
        num_cells = len(self.cellsNodes)

        # Get upper bound for number of edges; trim later.
        max_num_edges = num_local_nodes * num_cells
        self.edgesNodes = np.empty(max_num_edges, dtype=np.dtype((int,2)))
        self.edgesCells = [[] for k in xrange(max_num_edges)]
        self.cellsEdges = np.empty(num_cells, dtype=np.dtype((int,6)))

        # The (sorted) dictionary node_edges keeps track of how nodes and edges
        # are connected.
        # If  node_edges[(3,4)] == 17  is true, then the nodes (3,4) are
        # connected  by edge 17.
        edges = {}
        new_edge_gid = 0
        # Create edges.
        import itertools
        for cell_id in xrange(num_cells):
            # We're treating simplices so loop over all combinations of
            # local nodes.
            # Make sure cellNodes are sorted.
            self.cellsNodes[cell_id] = np.sort(self.cellsNodes[cell_id])
            for k, indices in enumerate(itertools.combinations(self.cellsNodes[cell_id], 2)):
                if indices in edges:
                    # edge already assigned
                    edge_gid = edges[indices]
                    self.edgesCells[edge_gid].append( cell_id )
                    self.cellsEdges[cell_id][k] = edge_gid
                else:
                    # add edge
                    self.edgesNodes[new_edge_gid] = indices
                    self.edgesCells[new_edge_gid].append( cell_id )
                    self.cellsEdges[cell_id][k] = new_edge_gid
                    edges[indices] = new_edge_gid
                    new_edge_gid += 1

        # trim edges
        self.edgesNodes = self.edgesNodes[:new_edge_gid]
        self.edgesCells = self.edgesCells[:new_edge_gid]

        # Create faces.
        max_num_faces = 4 * num_cells
        self.facesNodes = np.empty(max_num_faces, dtype=np.dtype((int,3)))
        self.facesEdges = np.empty(max_num_faces, dtype=np.dtype((int,3)))
        self.facesCells = [[] for k in xrange(max_num_faces)]
        self.edgesFaces = [[] for k in xrange(new_edge_gid)]
        self.cellsFaces = np.empty(num_cells, dtype=np.dtype((int,4)))
        # Loop over all elements.
        new_face_gid = 0
        registered_faces = {}
        for cell_id in xrange(num_cells):
            # Make sure cellNodes are sorted.
            self.cellsNodes[cell_id] = np.sort(self.cellsNodes[cell_id])
            for k in xrange(len(self.cellsNodes[cell_id])):
                # Remove the k-th element. This makes sure that the k-th
                # face is opposite of the k-th node. Useful later in
                # in construction of face normals.
                indices = tuple(self.cellsNodes[cell_id][:k]) \
                        + tuple(self.cellsNodes[cell_id][k+1:])
                if indices in registered_faces:
                    # Face already assigned, just register it with the
                    # current cell.
                    face_gid = registered_faces[indices]
                    self.facesCells[face_gid].append( cell_id )
                    self.cellsFaces[cell_id][k] = face_gid
                else:
                    # Add face.
                    # Make sure that facesNodes[k] and facesEdge[k] are
                    # coordinated in such a way that facesNodes[k][i]
                    # and facesEdge[k][i] are opposite in face k.
                    self.facesNodes[new_face_gid] = indices
                    # Register edges.
                    for kk in xrange(len(indices)):
                        # Note that node_tuple is also sorted, and thus
                        # is a key in the edges dictionary.
                        node_tuple = indices[:kk] + indices[kk+1:]
                        edge_id = edges[node_tuple]
                        self.edgesFaces[edge_id].append( new_face_gid )
                        self.facesEdges[new_face_gid][kk] = edge_id
                    # Register cells.
                    self.facesCells[new_face_gid].append( cell_id )
                    self.cellsFaces[cell_id][k] = new_face_gid
                    # Finalize.
                    registered_faces[indices] = new_face_gid
                    new_face_gid += 1

        # trim faces
        self.facesNodes = self.facesNodes[:new_face_gid]
        self.facesEdges = self.facesEdges[:new_face_gid]
        self.facesCells = self.facesCells[:new_face_gid]

        return
    # --------------------------------------------------------------------------
    def create_cell_circumcenters( self ):
        '''Computes the center of the circumsphere of each cell.
        '''
        import vtk
        num_cells = len(self.cellsNodes)
        self.cell_circumcenters = np.empty(num_cells, dtype=np.dtype((float,3)))
        for cell_id, cellNodes in enumerate(self.cellsNodes):
            x = self.nodes[cellNodes]
            vtk.vtkTetra.Circumsphere(x[0], x[1], x[2], x[3],
                                      self.cell_circumcenters[cell_id])
            ## http://www.cgafaq.info/wiki/Tetrahedron_Circumsphere
            #b = x[1] - x[0]
            #c = x[2] - x[0]
            #d = x[3] - x[0]

            #omega = (2.0 * np.dot( b, np.cross(c, d)))

            #if abs(omega) < 1.0e-10:
                #raise ZeroDivisionError( 'Tetrahedron is degenerate.' )
            #self.cell_circumcenters[cell_id] = x[0] + (   np.dot(b, b) * np.cross(c, d)
                            #+ np.dot(c, c) * np.cross(d, b)
                            #+ np.dot(d, d) * np.cross(b, c)
                          #) / omega

        return
    # --------------------------------------------------------------------------
    def _get_face_circumcenter(self, face_id):
        '''Computes the center of the circumcircle of each face.
        '''
        import vtk

        x = self.nodes[self.facesNodes[face_id]]
        # Project triangle to 2D.
        v = np.empty(3, dtype=np.dtype((float, 2)))
        vtk.vtkTriangle.ProjectTo2D(x[0], x[1], x[2],
                                    v[0], v[1], v[2])
        # Get the circumcenter in 2D.
        cc_2d = np.empty(2, dtype=float)
        vtk.vtkTriangle.Circumcircle(v[0], v[1], v[2], cc_2d)
        # Project back to 3D by using barycentric coordinates.
        bcoords = np.empty(3, dtype=float)
        vtk.vtkTriangle.BarycentricCoords(cc_2d, v[0], v[1], v[2], bcoords)
        return bcoords[0] * x[0] + bcoords[1] * x[1] + bcoords[2] * x[2]

        #a = x[0] - x[1]
        #b = x[1] - x[2]
        #c = x[2] - x[0]
        #w = np.cross(a, b)
        #omega = 2.0 * np.dot(w, w)
        #if abs(omega) < 1.0e-10:
            #raise ZeroDivisionError( 'The nodes don''t seem to form '
                                    #+ 'a proper triangle.' )
        #alpha = -np.dot(b, b) * np.dot(a, c) / omega
        #beta  = -np.dot(c, c) * np.dot(b, a) / omega
        #gamma = -np.dot(a, a) * np.dot(c, b) / omega
        #m = alpha * x[0] + beta * x[1] + gamma * x[2]

        ## Alternative implementation from
        ## https://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        #a = x[1] - x[0]
        #b = x[2] - x[0]
        #alpha = np.dot(a, a)
        #beta = np.dot(b, b)
        #w = np.cross(a, b)
        #omega = 2.0 * np.dot(w, w)
        #m = np.empty(3)
        #m[0] = x[0][0] + ((alpha * b[1] - beta * a[1]) * w[2]
                          #-(alpha * b[2] - beta * a[2]) * w[1]) / omega
        #m[1] = x[0][1] + ((alpha * b[2] - beta * a[2]) * w[0]
                          #-(alpha * b[0] - beta * a[0]) * w[2]) / omega
        #m[2] = x[0][2] + ((alpha * b[0] - beta * a[0]) * w[1]
                          #-(alpha * b[1] - beta * a[1]) * w[0]) / omega
        return
    # --------------------------------------------------------------------------
    def compute_control_volumes(self):
        '''Computes the control volumes of the mesh.'''

        if self.edgesNodes is None:
            self.create_adjacent_entities()

        # Get cell circumcenters.
        if self.cell_circumcenters is None:
            self.create_cell_circumcenters()

        # Compute covolumes and control volumes.
        num_nodes = len(self.nodes)
        self.control_volumes = np.zeros((num_nodes,1), dtype = float)
        for edge_id in xrange(len(self.edgesNodes)):
            edge_node_ids = self.edgesNodes[edge_id]
            edge = self.nodes[edge_node_ids[1]] \
                 - self.nodes[edge_node_ids[0]]
            edge_midpoint = 0.5 * ( self.nodes[edge_node_ids[0]]
                                  + self.nodes[edge_node_ids[1]])

            # 0.5 * alpha / edge_length = covolume.
            # This is chosen to avoid unnecessary calculation (such as
            # projecting onto the normalized edge and later multiplying
            # the aggregate by the edge length.
            alpha = 0.0
            for face_id in self.edgesFaces[edge_id]:
                # Make sure that the edge orientation is such that the covolume
                # contribution is positive if and only if the the vector
                # p[0]->p[1] is oriented like the face normal
                # (which is oriented cell[0]->cell[1]).
                # We need to make sure to gauge the edge orientation using
                # the face normal and one point *in* the face, e.g., the one
                # corner point that is not part of the edge. This makes sure
                # that certain nasty cases are properly dealt with, e.g., when
                # the edge midpoint does not sit in the covolume or that the
                # covolume orientation is clockwise while the corresponding
                # cells are oriented counter-clockwise.
                #
                # Find the edge in the list of edges of this face.
                # http://projects.scipy.org/numpy/ticket/1673
                edge_idx = np.nonzero(self.facesEdges[face_id] == edge_id)[0][0]
                # faceNodes and faceEdges need to be coordinates such that
                # the node faceNodes[face_id][k] and the edge
                # faceEdges[face_id][k] are opposing in the face face_id.
                opposing_point = self.nodes[self.facesNodes[face_id][edge_idx]]

                # Get the other point of one adjacent cell.
                # This involves:
                #   (a) Get the cell, get all its faces.
                #   (b) Find out which local index face_id is.
                # Then we rely on the data structure organized such that
                # cellsNodes[i][k] is opposite of cellsEdges[i][k] in
                # cell i.
                cell0 = self.facesCells[face_id][0]
                face0_idx = np.nonzero(self.cellsFaces[cell0] == face_id)[0][0]
                other0 = self.nodes[self.cellsNodes[cell0][face0_idx]]

                cc = self.cell_circumcenters[self.facesCells[face_id]]
                if len(cc) == 2:
                    # Get opposing point of the other cell.
                    cell1 = self.facesCells[face_id][1]
                    face1_idx = np.nonzero(self.cellsFaces[cell0] == face_id)[0][0]
                    other1 = self.nodes[self.cellsNodes[cell1][face1_idx]]
                    gauge = np.dot(edge, np.cross(other1 - other0,
                                                  opposing_point - edge_midpoint))
                    alpha += np.sign(gauge) \
                           * np.dot(edge, np.cross(cc[1] - edge_midpoint,
                                                cc[0] - edge_midpoint))
                elif len(cc) == 1:
                    # Each boundary face circumcenter is computed three times.
                    # Probably one could save a bit of CPU time by caching
                    # those.
                    face_cc = self._get_face_circumcenter(face_id)
                    gauge = np.dot(edge, np.cross(face_cc - other0,
                                                  opposing_point - edge_midpoint))
                    alpha += np.sign(gauge) \
                           * np.dot(edge, np.cross(face_cc - edge_midpoint,
                                                   cc[0] - edge_midpoint))
                else:
                    raise RuntimeError('A face should have either 1 or 2 adjacent cells.')

            # We add the pyramid volume
            #   covolume * 0.5*edgelength / 3,
            # which, given
            #   covolume = 0.5 * alpha / edge_length
            # is just
            #   0.25 * alpha / 3.
            self.control_volumes[edge_node_ids] += 0.25 * alpha / 3

        return
    # --------------------------------------------------------------------------
    def _compute_face_normals(self):
        # TODO VTK has ComputeNormal() for triangles, check
        # http://www.vtk.org/doc/nightly/html/classvtkTriangle.html

        # Precompute edge lengths.
        num_edges = len(self.edgesNodes)
        edge_lengths = np.empty(num_edges, dtype=float)
        for edge_id in xrange(num_edges):
            nodes = self.nodes[self.edgesNodes[edge_id]]
            edge_lengths[edge_id] = np.linalg.norm(nodes[1] - nodes[0])

        # Compute face normals. Do that in such a way that the
        # face normals points in the direction of the cell with the higher
        # cell ID.
        num_faces = len(self.facesNodes)
        face_normals = np.zeros(num_faces, dtype=np.dtype((float,3)))
        for cell_id, cellFaces in enumerate(self.cellsFaces):
            # Loop over the local faces.
            for k in xrange(4):
                face_id = cellFaces[k]
                # Compute the normal in the direction of the higher cell ID,
                # or if this is a boundary face, to the outside of the domain.
                neighbor_cell_ids = self.facesCells[face_id]
                if cell_id == neighbor_cell_ids[0]:
                    # The current cell is the one with the lower ID.
                    face_nodes = self.nodes[self.facesNodes[face_id]]
                    # Get "other" node (aka the one which is not in the current
                    # face).
                    other_node_id = self.cellsNodes[cell_id][k]
                    # Get any direction other_node -> face.
                    # As reference, any point in face can be taken, e.g.,
                    # face_nodes[0].
                    face_normals[face_id] = face_nodes[0] \
                                          - self.nodes[other_node_id]
                    if face_id == 2:
                        tmp = face_normals[face_id]
                    # Make it orthogonal to the face by doing Gram-Schmidt
                    # with the two edges of the face.
                    edge_id = self.facesEdges[face_id][0]
                    nodes = self.nodes[self.edgesNodes[edge_id]]
                    # No need to compute the norm of the first edge -- it's
                    # already here!
                    v0 = (nodes[1] - nodes[0]) / edge_lengths[edge_id]
                    edge_id = self.facesEdges[face_id][1]
                    nodes = self.nodes[self.edgesNodes[edge_id]]
                    v1 = nodes[1] - nodes[0]
                    v1 -= np.dot(v1, v0) * v0
                    v1 /= np.linalg.norm(v1)
                    face_normals[face_id] -= np.dot(face_normals[face_id], v0) * v0
                    face_normals[face_id] -= np.dot(face_normals[face_id], v1) * v1
                    # Normalization.
                    face_normals[face_id] /= np.linalg.norm(face_normals[face_id])

        return face_normals
    # --------------------------------------------------------------------------
    def show_edge(self, edge_id):
        '''Displays edge with covolume.'''
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # 3D axis aspect ratio isn't implemented in matplotlib yet (2012-02-21).
        #plt.axis('equal')

        if self.edgesNodes is None:
            self.create_adjacent_entities()

        edge_nodes = self.nodes[self.edgesNodes[edge_id]]

        # plot all adjacent cells
        col = 'k'
        for cell_id in self.edgesCells[edge_id]:
            for edge in self.cellsEdges[cell_id]:
                x = self.nodes[self.edgesNodes[edge]]
                ax.plot(x[:,0], x[:,1], x[:,2], col)

        # make clear which is the edge
        ax.plot(edge_nodes[:,0], edge_nodes[:,1], edge_nodes[:,2],
                color=col, linewidth=3.0 )

        # get cell circumcenters
        if self.cell_circumcenters is None:
            self.create_cell_circumcenters()
        cell_ccs = self.cell_circumcenters

        edge_midpoint = 0.5 * (edge_nodes[0] + edge_nodes[1])

        # plot covolume and highlight faces in matching colors
        num_local_faces = len(self.edgesFaces[edge_id])
        for k, face_id in enumerate(self.edgesFaces[edge_id]):
            # get rainbow color
            h = float(k) / num_local_faces
            hsv_face_col = np.array([[[h,1.0,1.0]]])
            col = mpl.colors.hsv_to_rgb(hsv_face_col)[0][0]

            # paint the face
            import mpl_toolkits.mplot3d as mpl3
            face_nodes = self.nodes[self.facesNodes[face_id]]
            tri = mpl3.art3d.Poly3DCollection( [face_nodes] )
            tri.set_color(mpl.colors.rgb2hex(col))
            #tri.set_alpha( 0.5 )
            ax.add_collection3d( tri )

            # mark face circumcenters
            face_cc = self._get_face_circumcenter(face_id)
            ax.plot([face_cc[0]], [face_cc[1]], [face_cc[2]],
                    marker='o', color=col)

            face_col = '0.7'
            ccs = cell_ccs[ self.facesCells[face_id] ]
            if len(ccs) == 2:
                tri = mpl3.art3d.Poly3DCollection([np.vstack((ccs, edge_midpoint))])
                tri.set_color(face_col)
                ax.add_collection3d( tri )
                ax.plot(ccs[:,0], ccs[:,1], ccs[:,2], color=col)
            elif len(ccs) == 1:
                tri = mpl3.art3d.Poly3DCollection([np.vstack((ccs[0], face_cc, edge_midpoint))])
                tri.set_color(face_col)
                ax.add_collection3d( tri )
                ax.plot([ccs[0][0],face_cc[0]],
                        [ccs[0][1],face_cc[1]],
                        [ccs[0][2],face_cc[2]],
                        color=col)
            else:
                raise RuntimeError('???')

        #ax.plot([edge_midpoint[0]], [edge_midpoint[1]], [edge_midpoint[2]], 'ro')

        # highlight cells
        #print self.edgesCells[edge_id]
        highlight_cells = [] #[3]
        col = 'r'
        for k in highlight_cells:
            cell_id = self.edgesCells[edge_id][k]
            ax.plot([cell_ccs[cell_id,0]], [cell_ccs[cell_id,1]], [cell_ccs[cell_id,2]],
                    color = col, marker='o')
            for edge in self.cellsEdges[cell_id]:
                x = self.nodes[self.edgesNodes[edge]]
                ax.plot(x[:,0], x[:,1], x[:,2], col, linestyle='dashed')

        plt.show()
        return
    # --------------------------------------------------------------------------
# ==============================================================================
