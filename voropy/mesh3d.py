# -*- coding: utf-8 -*-
# ==============================================================================
__all__ = ['mesh3d']

import numpy as np
from base import _base_mesh
# ==============================================================================
class mesh3d(_base_mesh):
    '''Class for handling three-dimensional tetrahedral meshes.
    '''
    # --------------------------------------------------------------------------
    def __init__(self, node_coords, cells):
        '''Initialization.
        '''
        super(mesh3d, self).__init__(node_coords, cells)
        self.node_coords = node_coords

        # Wait for Numpy 1.6.1 for this
        #     self.cells = np.array(cells, dtype=np.dtype([('nodes', (int, 3))]))
        # to work. Check out
        # http://stackoverflow.com/questions/9467547/how-to-properly-initialize-numpy-array-with-named-fields
        num_cells = len(cells)
        self.cells = np.empty(num_cells, dtype=np.dtype([('nodes', (int, 4))]))
        self.cells['nodes'] = cells

        self.edges = None
        self.faces = None

        self.cell_circumcenters = None
        self.cells_volume = None
        self.control_volumes = None
        return
    # --------------------------------------------------------------------------
    def create_cells_volume(self):
        '''Computes the volumes of the tetrahedra in the mesh.
        '''
        import vtk
        num_cells = len(self.cells['nodes'])
        self.cells_volume = np.empty(num_cells, dtype=float)
        for cell_id, cell in enumerate(self.cells):
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

            #self.cells_volume[cell_id] = abs( alpha ) / 6.0

            x = self.node_coords[cell['nodes']]
            self.cells_volume[cell_id] = \
                abs(vtk.vtkTetra.ComputeVolume(x[0], x[1], x[2], x[3]))
        return
    # --------------------------------------------------------------------------
    def create_adjacent_entities(self):
        '''Setup edge-node, edge-cell, edge-face, face-node, and face-cell
        relations.
        '''
        num_cells = len(self.cells['nodes'])

        # Get upper bound for number of edges; trim later.
        max_num_edges = 6 * num_cells
        dt = np.dtype([('nodes', (int, 2)),
                       ('faces', np.object),
                       ('cells', np.object)])
        # To create an array of empty lists, do what's described at
        # http://mail.scipy.org/pipermail/numpy-discussion/2009-November/046566.html
        self.edges = np.empty(max_num_edges, dt)
        filler = np.frompyfunc(lambda x: list(), 1, 1)
        self.edges['faces'] = filler(self.edges['faces'])
        self.edges['cells'] = filler(self.edges['cells'])


        # Extend the self.cells array by the keywords 'edges' and 'faces'.
        cells = self.cells['nodes']
        dt = np.dtype([('nodes', (int, 4)),
                       ('edges', (int, 6)),
                       ('faces', (int, 4))])
        self.cells = np.empty(len(cells), dtype=dt)
        self.cells['nodes'] = cells

        # The (sorted) dictionary node_edges keeps track of how nodes and edges
        # are connected.
        # If  node_edges[(3,4)] == 17  is true, then the nodes (3,4) are
        # connected  by edge 17.
        registered_edges = {}
        new_edge_gid = 0
        # Create edges.
        import itertools
        for cell_id, cell in enumerate(self.cells):
            # We're treating simplices so loop over all combinations of
            # local nodes.
            # Make sure cellNodes are sorted.
            self.cells['nodes'][cell_id] = np.sort(cell['nodes'])
            for k, indices in enumerate(itertools.combinations(cell['nodes'], 2)):
                if indices in registered_edges:
                    # edge already assigned
                    edge_gid = registered_edges[indices]
                    self.edges['cells'][edge_gid].append( cell_id )
                    self.cells['edges'][cell_id][k] = edge_gid
                else:
                    # add edge
                    self.edges['nodes'][new_edge_gid] = indices
                    self.edges['cells'][new_edge_gid].append( cell_id )
                    self.cells['edges'][cell_id][k] = new_edge_gid
                    registered_edges[indices] = new_edge_gid
                    new_edge_gid += 1

        # trim edges
        self.edges = self.edges[:new_edge_gid]

        # Create faces.
        max_num_faces = 4 * num_cells
        dt = np.dtype([('nodes', (int, 3)),
                       ('edges', (int, 3)),
                       ('cells', np.object)])
        self.faces = np.empty(max_num_faces, dt)
        self.faces['cells'] = filler(self.faces['cells'])

        # Loop over all elements.
        new_face_gid = 0
        registered_faces = {}
        for cell_id, cell in enumerate(self.cells):
            # Make sure cellNodes are sorted.
            self.cells['nodes'][cell_id] = np.sort(cell['nodes'])
            for k in xrange(4):
                # Remove the k-th element. This makes sure that the k-th
                # face is opposite of the k-th node. Useful later in
                # in construction of face normals.
                indices = tuple(cell['nodes'][:k]) \
                        + tuple(cell['nodes'][k+1:])
                if indices in registered_faces:
                    # Face already assigned, just register it with the
                    # current cell.
                    face_gid = registered_faces[indices]
                    self.faces['cells'][face_gid].append( cell_id )
                    self.cells['faces'][cell_id][k] = face_gid
                else:
                    # Add face.
                    # Make sure that facesNodes[k] and facesEdge[k] are
                    # coordinated in such a way that facesNodes[k][i]
                    # and facesEdge[k][i] are opposite in face k.
                    self.faces['nodes'][new_face_gid] = indices
                    # Register edges.
                    for kk in xrange(len(indices)):
                        # Note that node_tuple is also sorted, and thus
                        # is a key in the edges dictionary.
                        node_tuple = indices[:kk] + indices[kk+1:]
                        edge_id = registered_edges[node_tuple]
                        self.edges['faces'][edge_id].append( new_face_gid )
                        self.faces['edges'][new_face_gid][kk] = edge_id
                    # Register cells.
                    self.faces['cells'][new_face_gid].append( cell_id )
                    self.cells['faces'][cell_id][k] = new_face_gid
                    # Finalize.
                    registered_faces[indices] = new_face_gid
                    new_face_gid += 1

        # trim faces
        self.faces = self.faces[:new_face_gid]

        return
    # --------------------------------------------------------------------------
    def create_cell_circumcenters( self ):
        '''Computes the center of the circumsphere of each cell.
        '''
        import vtk
        num_cells = len(self.cells['nodes'])
        self.cell_circumcenters = np.empty(num_cells, dtype=np.dtype((float, 3)))
        for cell_id, cell in enumerate(self.cells):
            x = self.node_coords[cell['nodes']]
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
        '''Computes the center of the circumcircle of a given face.

        :params face_id: Face ID for which to compute circumcenter.
        :type face_id: int
        :returns circumcenter: Circumcenter of the face with given face ID.
        :type circumcenter: numpy.ndarray((float,3))
        '''
        import vtk

        x = self.node_coords[self.faces['nodes'][face_id]]
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
        #return
    # --------------------------------------------------------------------------
    def compute_control_volumes(self):
        '''Computes the control volumes of all nodes in mesh.'''

        if self.edges is None:
            self.create_adjacent_entities()

        # Get cell circumcenters.
        if self.cell_circumcenters is None:
            self.create_cell_circumcenters()

        # Compute covolumes and control volumes.
        num_nodes = len(self.node_coords)
        self.control_volumes = np.zeros((num_nodes, 1), dtype = float)
        for edge_id in xrange(len(self.edges['nodes'])):
            edge_node_ids = self.edges['nodes'][edge_id]
            edge = self.node_coords[edge_node_ids[1]] \
                 - self.node_coords[edge_node_ids[0]]
            edge_midpoint = 0.5 * ( self.node_coords[edge_node_ids[0]]
                                  + self.node_coords[edge_node_ids[1]])

            # 0.5 * alpha / edge_length = covolume.
            # This is chosen to avoid unnecessary calculation (such as
            # projecting onto the normalized edge and later multiplying
            # the aggregate by the edge length.
            alpha = 0.0
            for face_id in self.edges['faces'][edge_id]:
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
                edge_idx = np.nonzero(self.faces['edges'][face_id] == edge_id)[0][0]
                # faceNodes and faceEdges need to be coordinates such that
                # the node faceNodes[face_id][k] and the edge
                # faceEdges[face_id][k] are opposing in the face face_id.
                opposing_point = self.node_coords[self.faces['nodes'][face_id][edge_idx]]

                # Get the other point of one adjacent cell.
                # This involves:
                #   (a) Get the cell, get all its faces.
                #   (b) Find out which local index face_id is.
                # Then we rely on the data structure organized such that
                # cellsNodes[i][k] is opposite of cellsEdges[i][k] in
                # cell i.
                cell0 = self.faces['cells'][face_id][0]
                face0_idx = np.nonzero(self.cells['faces'][cell0] == face_id)[0][0]
                other0 = self.node_coords[self.cells['nodes'][cell0][face0_idx]]

                cc = self.cell_circumcenters[self.faces['cells'][face_id]]
                if len(cc) == 2:
                    # Get opposing point of the other cell.
                    cell1 = self.faces['cells'][face_id][1]
                    face1_idx = np.nonzero(self.cells['faces'][cell0] == face_id)[0][0]
                    other1 = self.node_coords[self.cells['nodes'][cell1][face1_idx]]
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
    def compute_face_normals(self):
        '''Compute the face normals, pointing either in the direction of the
        cell with larger GID (for interior faces), or towards the outside of
        the domain (for boundary faces).

        :returns face_normals: List of all face normals.
        :type face_normals: np.ndarray(num_faces, np.dtype((float, 3)))
        '''

        # TODO VTK has ComputeNormal() for triangles, check
        # http://www.vtk.org/doc/nightly/html/classvtkTriangle.html

        num_faces = len(self.faces['nodes'])
        face_normals = np.zeros(num_faces, dtype=np.dtype((float, 3)))
        for cell_id, cell in enumerate(self.cells):
            # Loop over the local faces.
            for k in xrange(4):
                face_id = cell['faces'][k]
                # Compute the normal in the direction of the higher cell ID,
                # or if this is a boundary face, to the outside of the domain.
                neighbor_cell_ids = self.faces['cells'][face_id]
                if cell_id == neighbor_cell_ids[0]:
                    # The current cell is the one with the lower ID.
                    # Compute the normal as a cross product.
                    face_nodes = self.node_coords[self.faces['nodes'][face_id]]
                    face_normals[face_id] = \
                        np.cross(face_nodes[1] - face_nodes[0],
                                 face_nodes[2] - face_nodes[0])
                    # Normalize.
                    face_normals[face_id] /= \
                        np.linalg.norm(face_normals[face_id])

                    # Make sure it points outwards.
                    other_node_id = self.cells['nodes'][cell_id][k]
                    other_node_coords = self.node_coords[other_node_id]
                    if np.dot(face_nodes[0] - other_node_coords,
                              face_normals[face_id]) < 0.0:
                        face_normals[face_id] *= -1

        return face_normals
    # --------------------------------------------------------------------------
    def show_edge(self, edge_id):
        '''Displays edge with covolume.

        :param edge_id: Edge ID for which to show the covolume.
        :type edge_id: int
        '''
        import matplotlib as mpl
        #from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # 3D axis aspect ratio isn't implemented in matplotlib yet (2012-02-21).
        #plt.axis('equal')

        if self.edges is None:
            self.create_adjacent_entities()

        edge_nodes = self.node_coords[self.edges['nodes'][edge_id]]

        # plot all adjacent cells
        col = 'k'
        for cell_id in self.edges['cells'][edge_id]:
            for edge in self.cells['edges'][cell_id]:
                x = self.node_coords[self.edges['nodes'][edge]]
                ax.plot(x[:, 0], x[:, 1], x[:, 2], col)

        # make clear which is the edge
        ax.plot(edge_nodes[:, 0], edge_nodes[:, 1], edge_nodes[:, 2],
                color=col, linewidth=3.0 )

        # get cell circumcenters
        if self.cell_circumcenters is None:
            self.create_cell_circumcenters()
        cell_ccs = self.cell_circumcenters

        edge_midpoint = 0.5 * (edge_nodes[0] + edge_nodes[1])

        # plot covolume and highlight faces in matching colors
        num_local_faces = len(self.edges['faces'][edge_id])
        for k, face_id in enumerate(self.edges['faces'][edge_id]):
            # get rainbow color
            h = float(k) / num_local_faces
            hsv_face_col = np.array([[[h, 1.0, 1.0]]])
            col = mpl.colors.hsv_to_rgb(hsv_face_col)[0][0]

            # paint the face
            import mpl_toolkits.mplot3d as mpl3
            face_nodes = self.node_coords[self.faces['nodes'][face_id]]
            tri = mpl3.art3d.Poly3DCollection( [face_nodes] )
            tri.set_color(mpl.colors.rgb2hex(col))
            #tri.set_alpha( 0.5 )
            ax.add_collection3d( tri )

            # mark face circumcenters
            face_cc = self._get_face_circumcenter(face_id)
            ax.plot([face_cc[0]], [face_cc[1]], [face_cc[2]],
                    marker='o', color=col)

            face_col = '0.7'
            ccs = cell_ccs[ self.faces['cells'][face_id] ]
            if len(ccs) == 2:
                tri = mpl3.art3d.Poly3DCollection([np.vstack((ccs, edge_midpoint))])
                tri.set_color(face_col)
                ax.add_collection3d( tri )
                ax.plot(ccs[:, 0], ccs[:, 1], ccs[:, 2], color=col)
            elif len(ccs) == 1:
                tri = mpl3.art3d.Poly3DCollection([np.vstack((ccs[0], face_cc, edge_midpoint))])
                tri.set_color(face_col)
                ax.add_collection3d( tri )
                ax.plot([ccs[0][0], face_cc[0]],
                        [ccs[0][1], face_cc[1]],
                        [ccs[0][2], face_cc[2]],
                        color=col)
            else:
                raise RuntimeError('???')

        #ax.plot([edge_midpoint[0]], [edge_midpoint[1]], [edge_midpoint[2]], 'ro')

        # highlight cells
        #print self.edges['cells'][edge_id]
        highlight_cells = [] #[3]
        col = 'r'
        for k in highlight_cells:
            cell_id = self.edges['cells'][edge_id][k]
            ax.plot([cell_ccs[cell_id, 0]],
                    [cell_ccs[cell_id, 1]],
                    [cell_ccs[cell_id, 2]],
                    color = col, marker='o')
            for edge in self.cells['edges'][cell_id]:
                x = self.node_coords[self.edges['nodes'][edge]]
                ax.plot(x[:, 0], x[:, 1], x[:, 2], col, linestyle='dashed')

        plt.show()
        return
    # --------------------------------------------------------------------------
# ==============================================================================
