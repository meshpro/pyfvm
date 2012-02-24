# -*- coding: utf-8 -*-
"""Module for I/O of unstructured grids to/from various file formats.
"""
# ==============================================================================
import vtk
import os, sys
import numpy as np
import mesh
# ==============================================================================
def read( file_name, timestep=None ):
    '''Reads an FEM mesh from an Exodus file.'''
    extension = os.path.splitext( file_name )[1]

    # setup the reader
    if extension == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
        return _read_vtk_grid( reader, file_name )

    elif extension == ".vtk":
        reader = vtk.vtkUnstructuredGridReader()
        return _read_vtk_grid( reader, file_name )

    elif extension in [ ".ex2", ".exo", ".e" ]:
        reader = vtk.vtkExodusIIReader()
        return _read_exodusii_grid( reader, file_name, timestep=timestep )

    else:
        sys.exit( "Unknown file type \"%s\"." % filename )
# ==============================================================================
def _read_vtk_grid( reader, file_name ):
    '''Uses a vtkReader to return a vtkUnstructuredGrid.
    '''
    reader.SetFileName( file_name )
    reader.Update()

    return reader.GetOutput()
# ==============================================================================
def _read_exodus_grid( reader, file_name ):
    '''Uses a vtkExodusIIReader to return a vtkUnstructuredGrid.
    '''
    reader.SetFileName( file_name )

    # Create Exodus metadata that can be used later when writing the file.
    reader.ExodusModelMetadataOn()

    # Fetch metadata.
    reader.UpdateInformation()

    # Make sure the point fields are read during Update().
    for k in xrange( reader.GetNumberOfPointArrays() ):
        arr_name = reader.GetPointArrayName( k )
        reader.SetPointArrayStatus( arr_name, 1 )

    # Read the file.
    reader.Update()

    return reader.GetOutput()
# ==============================================================================
def _read_exodusii_grid( reader, file_name, timestep=None ):
    '''Uses a vtkExodusIIReader to return a vtkUnstructuredGrid.
    '''
    reader.SetFileName( file_name )

    # Fetch metadata.
    reader.UpdateInformation()

    # Set time step to read.
    if timestep:
        reader.SetTimeStep( timestep )

    # Make sure the point fields are read during Update().
    for k in xrange( reader.GetNumberOfPointResultArrays() ):
        arr_name = reader.GetPointResultArrayName( k )
        reader.SetPointResultArrayStatus( arr_name, 1 )

    # Read the file.
    reader.Update()
    out = reader.GetOutput()

    # Loop through the blocks and search for a vtkUnstructuredGrid.
    vtk_mesh = []
    for i in xrange( out.GetNumberOfBlocks() ):
        #print out.GetMetaData( i ).Get( vtk.vtkCompositeDataSet.NAME() )
        blk = out.GetBlock( i )
        for j in xrange( blk.GetNumberOfBlocks() ):
            #print '  ' + blk.GetMetaData( j ).Get( vtk.vtkCompositeDataSet.NAME() )
            sub_block = blk.GetBlock( j )
            if sub_block.IsA( "vtkUnstructuredGrid" ):
                vtk_mesh.append( sub_block )

    if len(vtk_mesh) == 0:
        raise IOError( "No 'vtkUnstructuredGrid' found!" )
    elif len(vtk_mesh) > 1:
        raise IOError( "More than one 'vtkUnstructuredGrid' found!" )

    # Cut off trailing "_".
    for k in xrange( vtk_mesh[0].GetPointData().GetNumberOfArrays() ):
        array = vtk_mesh[0].GetPointData().GetArray(k)
        array_name = array.GetName()
        if array_name[-1] == "_":
            array.SetName( array_name[0:-1] )

    return vtk_mesh[0]
# ==============================================================================
def read_mesh(filename, timestep=None):
    '''Reads an FEM mesh with added data.
    '''
    vtk_mesh = read( filename, timestep=timestep )

    # extract points, elems, elemtypes
    points = _extract_points( vtk_mesh )
    cellsNodes = _extract_cellsNodes ( vtk_mesh )
    psi, A = _extract_values( vtk_mesh )

    num_points = len( points )
    if num_points != len( psi ) or num_points != len( A ):
        raise Exception, "The number of points (%d) must equal to" + \
                         "the number of values psi (%d) and A (%d)."  % (num_points, len(psi), len(A) )
    
    # gather the field data
    field_data = _read_field_data( vtk_mesh )

    if len(cellsNodes[0]) == 3: # 2D
        if all(points[:,2] == 0.0):
           # flat mesh
           assert all(A[:,2] == 0.0)
           from mesh2d import Mesh2D
           return Mesh2D( points[:,:2], cellsNodes ), psi, A[:,:2], field_data
        else:
            # shell mesh
            from mesh2d_shell import Mesh2DShell
            return Mesh2DShell( points, cellsNodes ), psi, A, field_data
    elif len(cellsNodes[0]) == 4: # 3D
        from mesh3d import Mesh3D
        return Mesh3D( points, cellsNodes ), psi, A, field_data
    else:
        raise RuntimeError('Unknonw mesh type.')
# ==============================================================================
def _extract_points( vtk_mesh ):

    num_points = vtk_mesh.GetNumberOfPoints()

    # --------------------------------------------------------------------------
    # Find out which points sit on the boundary.
    # TODO This is part of the code is *ugly*: come up with something better,
    #      Read VTK documentation on how to identify boundary points?
    # Determine if a point is a boundary point.
    # Transform vtkMesh into vtkPolyData.
    surfaceFilter = vtk.vtkDataSetSurfaceFilter()
    surfaceFilter.SetInput( vtk_mesh )
    surfaceFilter.Update()

    # filter out the boundary edges
    pEdges = vtk.vtkFeatureEdges()
    pEdges.SetInput( surfaceFilter.GetOutput() )
    pEdges.BoundaryEdgesOn()
    pEdges.FeatureEdgesOff()
    pEdges.NonManifoldEdgesOff()
    pEdges.ManifoldEdgesOff()
    pEdges.Update()

    poly = pEdges.GetOutput()

    #is_on_boundary = np.empty( num_points, dtype=bool )
    #for k in range( poly.GetNumberOfPoints() ):
        #x = poly.GetPoint( k )
        #ptId = vtk_mesh.FindPoint( x )
        #is_on_boundary[ ptId ] = True
    # --------------------------------------------------------------------------
    # construct the points list
    points = np.empty( num_points, np.dtype((float, 3)))
    for k in range(num_points):
        points[k] = np.array( vtk_mesh.GetPoint( k ) )
    # --------------------------------------------------------------------------

    return points
# ==============================================================================
def _extract_cellsNodes( vtk_mesh ):

    num_cells = vtk_mesh.GetNumberOfCells()
    num_local_nodes = vtk_mesh.GetCell(0).GetNumberOfPoints()
    cellsNodes = np.empty(num_cells, dtype = np.dtype((int,num_local_nodes)))

    for k in xrange( vtk_mesh.GetNumberOfCells() ):
        cell = vtk_mesh.GetCell( k )

        # gather up the points
        num_points = cell.GetNumberOfPoints()
        for l in xrange( num_points ):
            cellsNodes[k][l] = cell.GetPointId( l )

    return cellsNodes
# ==============================================================================
def _extract_values( vtk_data ):
    '''Extract a complex-valued vector out of a VTK data set.
    '''

    arrays = []
    for k in xrange( vtk_data.GetPointData().GetNumberOfArrays() ):
        arrays.append( vtk_data.GetPointData().GetArray(k) )

    # Go through all arrays, fetch psi and A.
    for array in arrays:
        # read the array
        arrayName = array.GetName()
        if arrayName == 'psi':
            assert array.GetNumberOfComponents() == 2
            num_entries = array.GetNumberOfTuples()
            # Create the complex array.
            z = np.empty( (num_entries,1), dtype = complex )
            for k in xrange( num_entries ):
                z[k] = complex( array.GetComponent( k, 0 ),
                                array.GetComponent( k, 1 )
                              )
        elif arrayName == 'A':
            assert array.GetNumberOfComponents() == 3
            num_entries = array.GetNumberOfTuples()
            # Create the complex array.
            A = np.empty(num_entries, dtype=np.dtype((float,3)) )
            for k in xrange( num_entries ):
                A[k] = [ array.GetComponent( k, 0 ),
                         array.GetComponent( k, 1 ),
                         array.GetComponent( k, 2 )
                       ]
        else:
            msg = "Unexpected array \"%s\". Skipping." % arrayName

    return z, A
# ==============================================================================
def _create_int_field_data ( arr, name ):

    field_data = vtk.vtkIntArray()

    field_data.SetName ( name )

    # fill the field
    for x in arr:
        field_data.InsertNextValue( x )

    return field_data
# ==============================================================================
def _read_field_data( vtk_data ):
    '''Gather field data.'''

    vtk_field_data = vtk_data.GetFieldData()
    num_arrays = vtk_field_data.GetNumberOfArrays()

    field_data = {}
    for k in xrange( num_arrays ):
        array  = vtk_field_data.GetArray(k)
        name   = array.GetName()
        num_values = array.GetDataSize()
        if num_values == 1:
            values = array.GetValue( k )
        else:
            values = np.zeros( num_values )
            for i in xrange( num_values ):
                values[i] = array.GetValue(i)
        field_data[ name ] = values

    return field_data
# ==============================================================================
