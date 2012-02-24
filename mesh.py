# -*- coding: utf-8 -*-
# ==============================================================================
import numpy as np
# ==============================================================================
class Mesh:
    # --------------------------------------------------------------------------
    def __init__(self,
                 nodes,
                 cellsNodes
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

        self.nodes = nodes
        self.cellsNodes = cellsNodes
        self.cellsVolume = None
        self.vtk_mesh = None
    # --------------------------------------------------------------------------
    def write( self,
               filename,
               extra_arrays = None
             ):
        '''Writes data together with the mesh to a file.
        '''
        import os
        import vtk

        if self.vtk_mesh is None:
            self.vtk_mesh = self._generate_vtk_mesh(self.nodes, self.cellsNodes)

        # add arrays
        if extra_arrays:
            for key, value in extra_arrays.iteritems():
                self.vtk_mesh.GetPointData().AddArray(
                    self._create_vtkdoublearray(value, key))

        extension = os.path.splitext(filename)[1]
        if extension == ".vtu": # VTK XML format
            writer = vtk.vtkXMLUnstructuredGridWriter()
        elif extension == ".pvtu": # parallel VTK XML format
            writer = vtk.vtkXMLPUnstructuredGridWriter()
        elif extension == ".vtk": # classical VTK format
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileTypeToASCII()
        elif extension in [ ".ex2", ".exo", ".e" ]: # Exodus II format
            writer = vtk.vtkExodusIIWriter()
            # If the mesh contains vtkModelData information, make use of it
            # and write out all time steps.
            writer.WriteAllTimeStepsOn()
        else:
            raise IOError( "Unknown file type \"%s\"." % filename )

        writer.SetFileName( filename )

        writer.SetInput( self.vtk_mesh )

        writer.Write()

        return
    # --------------------------------------------------------------------------
    def _generate_vtk_mesh(self, points, cellsNodes, X = None, name = None):
        import vtk
        mesh = vtk.vtkUnstructuredGrid()

        # set points
        vtk_points = vtk.vtkPoints()
        for point in points:
            vtk_points.InsertNextPoint( point )
        mesh.SetPoints( vtk_points )

        # set cells
        for cellNodes in cellsNodes:
            pts = vtk.vtkIdList()
            num_local_nodes = len(cellNodes)
            pts.SetNumberOfIds(num_local_nodes)
            # get the connectivity for this element
            k = 0
            # TODO insert the whole thing at once?
            for node_index in cellNodes:
                pts.InsertId(k, node_index)
                k += 1
            if num_local_nodes == 3:
                element_type = vtk.VTK_TRIANGLE
            elif num_local_nodes == 4:
                element_type = vtk.VTK_TETRA
            else:
                raise ValueError('Unknown element.')
            mesh.InsertNextCell(element_type, pts)

        # set values
        if X is not None:
            mesh.GetPointData().AddArray(self._create_vtkdoublearray(X, name))

        return mesh
    # --------------------------------------------------------------------------
    def _create_vtkdoublearray(self, X, name):
        import vtk

        scalars0 = vtk.vtkDoubleArray()
        scalars0.SetName(name)

        if isinstance( X, float ):
            scalars0.SetNumberOfComponents( 1 )
            scalars0.InsertNextValue( X )
        elif (len( X.shape ) == 1 or X.shape[1] == 1) and X.dtype==float:
            # real-valued array
            scalars0.SetNumberOfComponents( 1 )
            for x in X:
                scalars0.InsertNextValue( x )

        elif (len( X.shape ) == 1 or X.shape[1] == 1) and X.dtype==complex:
            # complex-valued array
            scalars0.SetNumberOfComponents( 2 )
            for x in X:
                scalars0.InsertNextValue( x.real )
                scalars0.InsertNextValue( x.imag )

        elif len( X.shape ) == 2 and X.dtype==float: # 2D float field
            m, n = X.shape
            scalars0.SetNumberOfComponents( n )
            for j in range(m):
                for i in range(n):
                    scalars0.InsertNextValue( X[j, i] )

        elif len( X.shape ) == 2 and X.dtype==complex: # 2D complex field
            scalars0.SetNumberOfComponents( 2 )
            m, n = X.shape
            for j in range(n):
                for i in range(m):
                    scalars0.InsertNextValue( X[j, i].real )
                    scalars0.InsertNextValue( X[j, i].imag )

        elif len( X.shape ) == 3: # vector values
            m, n, d = X.shape
            if X.dtype==complex:
                raise "Can't handle complex-valued vector fields."
            if d != 3:
                raise "Can only deal with 3-dimensional vector fields."
            scalars0.SetNumberOfComponents( 3 )
            for j in range( n ):
                for i in range( m ):
                    for k in range( 3 ):
                        scalars0.InsertNextValue( X[i,j,k] )

        else:
            raise ValueError( "Don't know what to do with array." )

        return scalars0
    # --------------------------------------------------------------------------
    def recreate_cells_with_qhull(self):
        import scipy.spatial

        # Create a Delaunay triangulation of the given points.
        delaunay = scipy.spatial.Delaunay(self.nodes)
        # Use the new cells.
        self.cellsNodes = delaunay.vertices

        return
    # --------------------------------------------------------------------------
# ==============================================================================
