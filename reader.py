# -*- coding: utf-8 -*-
"""Module for I/O of unstructured grids to/from various file formats.
"""
# ==============================================================================
__all__ = ['read']
# ==============================================================================
import vtk
import os
import numpy as np
# ==============================================================================
def read(filename, timestep=None):
    '''Reads an unstructured mesh with added data.
    '''
    vtk_mesh = _read_mesh(filename, timestep=timestep)

    # read points, cells, point data, field data
    points = _read_points( vtk_mesh )
    cellsNodes = _read_cellsNodes ( vtk_mesh )
    point_data = _read_point_data( vtk_mesh )
    field_data = _read_field_data( vtk_mesh )

    if len(cellsNodes[0]) == 3: # 2D
        if all(points[:,2] == 0.0):
           # Flat mesh.
           # Check if there's three-dimensional point data that can be cut.
           # Don't use iteritems() here as we want to be able to
           # set the value in the loop.
           for key, value in point_data.items():
               if value.shape[1] == 3 and all(value[:,2] == 0.0):
                   point_data[key] = value[:,:2]
           from voropy import mesh2d
           return mesh2d(points[:,:2], cellsNodes), point_data, field_data
        else:
            # shell mesh
            from voropy import mesh2d_shell
            return mesh2Dshell( points, cellsNodes ), point_data, field_data
    elif len(cellsNodes[0]) == 4: # 3D
        from voropy import mesh3d
        return mesh3d( points, cellsNodes ), point_data, field_data
    else:
        raise RuntimeError('Unknown mesh type.')

    return
# ==============================================================================
def _read_mesh(file_name, timestep=None):
    '''Reads an unstructured mesh an a file.'''
    extension = os.path.splitext( file_name )[1]

    # setup the reader
    if extension == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
        return _read_vtk_mesh( reader, file_name )
    elif extension == ".vtk":
        reader = vtk.vtkUnstructuredGridReader()
        return _read_vtk_mesh( reader, file_name )
    elif extension in [ ".ex2", ".exo", ".e" ]:
        reader = vtk.vtkExodusIIReader()
        return _read_exodusii_mesh( reader, file_name, timestep=timestep )
    else:
        raise RuntimeError( "Unknown file type \"%s\"." % filename )

    return
# ==============================================================================
def _read_vtk_mesh( reader, file_name ):
    '''Uses a vtkReader to return a vtkUnstructuredGrid.
    '''
    reader.SetFileName( file_name )
    reader.Update()

    return reader.GetOutput()
# ==============================================================================
#def _read_exodus_grid( reader, file_name ):
    #'''Uses a vtkExodusIIReader to return a vtkUnstructuredGrid.
    #'''
    #reader.SetFileName( file_name )

    ## Create Exodus metadata that can be used later when writing the file.
    #reader.ExodusModelMetadataOn()

    ## Fetch metadata.
    #reader.UpdateInformation()

    ## Make sure the point fields are read during Update().
    #for k in xrange( reader.GetNumberOfPointArrays() ):
        #arr_name = reader.GetPointArrayName( k )
        #reader.SetPointArrayStatus( arr_name, 1 )

    ## Read the file.
    #reader.Update()

    #return reader.GetOutput()
# ==============================================================================
def _read_exodusii_mesh( reader, file_name, timestep=None ):
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
def _read_points( vtk_mesh ):

    num_points = vtk_mesh.GetNumberOfPoints()

    # construct the points list
    points = np.empty( num_points, np.dtype((float, 3)))
    for k in range(num_points):
        points[k] = np.array( vtk_mesh.GetPoint( k ) )

    return points
# ==============================================================================
def _read_cellsNodes( vtk_mesh ):

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
def _read_point_data( vtk_data ):
    '''Extract point data from a VTK data set.
    '''
    arrays = []
    for k in xrange( vtk_data.GetPointData().GetNumberOfArrays() ):
        arrays.append( vtk_data.GetPointData().GetArray(k) )

    # Go through all arrays, fetch psi and A.
    out = {}
    for array in arrays:
        # read the array
        arrayName = array.GetName()
        num_entries = array.GetNumberOfTuples()
        num_components = array.GetNumberOfComponents()
        out[arrayName] = np.empty((num_entries, num_components))
        for k in xrange( num_entries ):
            for i in xrange( num_components ):
                out[arrayName][k][i] = array.GetComponent(k, i)

    return out
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
