# -*- coding: utf-8 -*-
# ==============================================================================
__all__ = []

# ==============================================================================
class _base_mesh(object):
    # --------------------------------------------------------------------------
    def __init__(self,
                 nodes,
                 cells_nodes
                 ):
        return
    # --------------------------------------------------------------------------
    def write(self,
              filename,
              point_data = None,
              field_data = None
              ):
        '''Writes mesh together with data to a file.

        :params filename: File to write to.
        :type filename: str

        :params point_data: Named additional point data to write to the file.
        :type point_data: dict
        '''
        import os

        vtk_mesh = self._generate_vtk_mesh(self.node_coords, self.cells['nodes'])

        # add point data
        if point_data:
            for key, value in point_data.iteritems():
                vtk_mesh.GetPointData() \
                        .AddArray(_create_vtkarray(value, key))

        # add field data
        if field_data:
            for key, value in field_data.iteritems():
                vtk_mesh.GetFieldData() \
                        .AddArray(_create_vtkarray(value, key))

        import re
        extension = os.path.splitext(filename)[1]
        if extension == '.vtu': # VTK XML format
            from vtk import vtkXMLUnstructuredGridWriter
            writer = vtkXMLUnstructuredGridWriter()
        elif extension == '.pvtu': # parallel VTK XML format
            from vtk import vtkXMLPUnstructuredGridWriter
            writer = vtkXMLPUnstructuredGridWriter()
        elif extension == '.vtk': # classical VTK format
            from vtk import vtkUnstructuredGridWriter
            writer = vtkUnstructuredGridWriter()
            writer.SetFileTypeToASCII()
        elif extension in [ '.ex2', '.exo', '.e' ]: # Exodus II format
            from vtk import vtkExodusIIWriter
            writer = vtkExodusIIWriter()
            # If the mesh contains vtkModelData information, make use of it
            # and write out all time steps.
            writer.WriteAllTimeStepsOn()
        elif re.match('[^\.]*\.e\.\d+\.\d+', filename):
            # TODO handle parallel I/O with vtkPExodusIIWriter
            from vtk import vtkExodusIIWriter
            writer = vtkExodusIIWriter()
            # If the mesh contains vtkModelData information, make use of it
            # and write out all time steps.
            writer.WriteAllTimeStepsOn()
        else:
            raise IOError( 'Unknown file type \'%s\'.' % filename )

        writer.SetFileName( filename )

        writer.SetInput( vtk_mesh )

        writer.Write()

        return
    # --------------------------------------------------------------------------
    def _generate_vtk_mesh(self, points, cellsNodes):
        from vtk import vtkUnstructuredGrid, VTK_TRIANGLE, VTK_TETRA, vtkIdList, vtkPoints
        mesh = vtkUnstructuredGrid()

        # set points
        vtk_points = vtkPoints()
        if len(points[0]) == 2:
            cell_type = VTK_TRIANGLE
            for point in points:
                vtk_points.InsertNextPoint(point[0], point[1], 0.0)
        elif len(points[0]) == 3:
            cell_type = VTK_TETRA
            for point in points:
                vtk_points.InsertNextPoint(point[0], point[1], point[2])
        else:
            raise RuntimeError('???')
        mesh.SetPoints( vtk_points )

        # set cells
        for cellNodes in cellsNodes:
            pts = vtkIdList()
            num_local_nodes = len(cellNodes)
            pts.SetNumberOfIds(num_local_nodes)
            # get the connectivity for this element
            for k, node_index in enumerate(cellNodes):
                pts.InsertId(k, node_index)
            mesh.InsertNextCell(cell_type, pts)

        return mesh
    # --------------------------------------------------------------------------
    def recreate_cells_with_qhull(self):
        '''Remesh using scipy.spatial.Delaunay.
        '''
        import scipy.spatial

        # Create a Delaunay triangulation of the given points.
        delaunay = scipy.spatial.Delaunay(self.nodes)
        # Use the new cells.
        self.cells['nodes'] = delaunay.vertices

        return
    # --------------------------------------------------------------------------
# ==============================================================================
def _create_vtkarray(X, name):
    from vtk import vtkDoubleArray, vtkIntArray, vtkStringArray

    # This could be a lot more fine-grained:
    # vtkLongLongArray, vtkFloatArray,...
    if isinstance(X, int):
        array = vtkIntArray()
        array.SetNumberOfComponents( 1 )
        array.InsertNextValue( X )
    elif isinstance(X, float):
        array = vtkDoubleArray()
        array.SetNumberOfComponents( 1 )
        array.InsertNextValue( X )
    elif isinstance(X, str):
        array = vtkStringArray()
        array.SetNumberOfComponents( 1 )
        for letter in X:
            array.InsertNextValue( letter )
    elif (len(X.shape) == 1 or X.shape[1] == 1) and X.dtype==float:
        # real-valued array
        array = vtkDoubleArray()
        array.SetNumberOfComponents( 1 )
        for x in X:
            array.InsertNextValue( x )

    elif (len( X.shape ) == 1 or X.shape[1] == 1) and X.dtype==complex:
        # complex-valued array
        array = vtkDoubleArray()
        array.SetNumberOfComponents( 2 )
        for x in X:
            array.InsertNextValue( x.real )
            array.InsertNextValue( x.imag )

    elif len( X.shape ) == 2 and X.dtype==float: # 2D float field
        array = vtkDoubleArray()
        m, n = X.shape
        array.SetNumberOfComponents( n )
        for j in range(m):
            for i in range(n):
                array.InsertNextValue( X[j, i] )

    elif len( X.shape ) == 2 and X.dtype==complex: # 2D complex field
        array = vtkDoubleArray()
        array.SetNumberOfComponents( 2 )
        m, n = X.shape
        for j in range(n):
            for i in range(m):
                array.InsertNextValue( X[j, i].real )
                array.InsertNextValue( X[j, i].imag )

    elif len( X.shape ) == 3: # vector values
        array = vtkDoubleArray()
        m, n, d = X.shape
        if X.dtype == complex:
            raise RuntimeError('Can''t handle complex-valued vector fields.')
        if d != 3:
            raise RuntimeError('Can only deal with 3-dimensional vector fields.')
        array.SetNumberOfComponents( 3 )
        for j in range( n ):
            for i in range( m ):
                for k in range( 3 ):
                    array.InsertNextValue(X[i, j, k])

    else:
        raise ValueError('Don''t know what to do with array.')

    array.SetName(name)

    return array
# ==============================================================================
