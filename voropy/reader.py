# -*- coding: utf-8 -*-
'''Module for reading unstructured grids (and related data) from various
file formats.

.. moduleauthor:: Nico Schloemer <nico.schloemer@gmail.com>

'''
# ==============================================================================
__all__ = ['read']
# ==============================================================================
import os
import numpy as np
# ==============================================================================
def read(filenames, timestep=None):
    '''Reads an unstructured mesh with added data.

    :param filenames: The files to read from.
    :type filenames: str
    :param timestep: Time step to read from, in case of an Exodus input mesh.
    :type timestep: int, optional
    :returns mesh{2,3}d: The mesh data.
    :returns point_data: Point data read from file.
    :type point_data: dict
    :returns field_data: Field data read from file.
    :type field_data: dict
    '''
    vtk_mesh = read_raw(filenames, timestep)

    # Explicitly extract points, cells, point data, field data
    points = _read_points( vtk_mesh )
    cells_nodes = _read_cells_nodes ( vtk_mesh )
    point_data = _read_point_data( vtk_mesh )
    field_data = _read_field_data( vtk_mesh )

    if len(cells_nodes[0]) == 3 and all(points[:, 2] == 0.0):
        # Flat mesh.
        # Check if there's three-dimensional point data that can be cut.
        # Don't use iteritems() here as we want to be able to
        # set the value in the loop.
        for key, value in point_data.items():
            if value.shape[1] == 3 and all(value[:, 2] == 0.0):
                point_data[key] = value[:, :2]
        from voropy import mesh2d
        return mesh2d(points[:, :2], cells_nodes), point_data, field_data
    elif len(cells_nodes[0]) == 4: # 3D
        from voropy import mesh3d
        return mesh3d(points, cells_nodes), point_data, field_data
    else:
        raise RuntimeError('Unknown mesh type.')

    return
# ==============================================================================
def read_raw(filenames, timestep):
    '''Read the data in raw VTK format.'''
    if isinstance(filenames, (list, tuple)) and len(filenames)==1:
        filenames = filenames[0]

    if isinstance(filenames, basestring):
        filename = filenames
        # serial files
        extension = os.path.splitext(filename)[1]

        import re
        # setup the reader
        # TODO Most readers have CanReadFile() -- use that.
        if extension == '.vtu':
            from vtk import vtkXMLUnstructuredGridReader
            reader = vtkXMLUnstructuredGridReader()
            vtk_mesh = _read_vtk_mesh(reader, filename)
        elif extension == '.vtk':
            from vtk import vtkUnstructuredGridReader
            reader = vtkUnstructuredGridReader()
            vtk_mesh = _read_vtk_mesh(reader, filename)
        elif extension in [ '.ex2', '.exo', '.e' ]:
            from vtk import vtkExodusIIReader
            reader = vtkExodusIIReader()
            reader.SetFileName( filename )
            vtk_mesh = _read_exodusii_mesh(reader, timestep=timestep)
            #print time_values
        elif re.match('[^\.]*\.e\.\d+\.\d+', filename):
            # Parallel Exodus files.
            # TODO handle with vtkPExodusIIReader
            from vtk import vtkExodusIIReader
            reader = vtkExodusIIReader()
            reader.SetFileName( filenames[0] )
            vtk_mesh = _read_exodusii_mesh(reader, timestep=timestep)
        else:
            raise RuntimeError( 'Unknown file type \'%s\'.' % filename )
    else:
        # Parallel files.
        # Assume Exodus format as we don't know anything else yet.
        from vtk import vtkPExodusIIReader
        # TODO Guess the file pattern or whatever.
        reader = vtkPExodusIIReader()
        reader.SetFileNames( filenames )
        vtk_mesh = _read_exodusii_mesh(reader, filename, timestep=timestep)

    return vtk_mesh
# ==============================================================================
def _read_vtk_mesh( reader, file_name ):
    '''Uses a vtkReader to return a vtkUnstructuredGrid.
    '''
    reader.SetFileName( file_name )
    reader.Update()

    return reader.GetOutput()
# ==============================================================================
#def _read_exodus_mesh( reader, file_name ):
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
def _read_exodusii_mesh( reader, timestep=None ):
    '''Uses a vtkExodusIIReader to return a vtkUnstructuredGrid.
    '''
    # Fetch metadata.
    reader.UpdateInformation()

    # Set time step to read.
    if timestep:
        reader.SetTimeStep( timestep )

    # Make sure the point fields are read during Update().
    for k in xrange( reader.GetNumberOfPointResultArrays() ):
        arr_name = reader.GetPointResultArrayName( k )
        reader.SetPointResultArrayStatus( arr_name, 1 )

    # Make sure all field data is read.
    for k in xrange( reader.GetNumberOfGlobalResultArrays() ):
        arr_name = reader.GetGlobalResultArrayName( k )
        reader.SetGlobalResultArrayStatus( arr_name, 1 )

    # Read the file.
    reader.Update()
    out = reader.GetOutput()

    # Loop through the blocks and search for a vtkUnstructuredGrid.
    vtk_mesh = []
    for i in xrange( out.GetNumberOfBlocks() ):
        blk = out.GetBlock( i )
        for j in xrange( blk.GetNumberOfBlocks() ):
            sub_block = blk.GetBlock( j )
            if sub_block.IsA( 'vtkUnstructuredGrid' ):
                vtk_mesh.append( sub_block )

    if len(vtk_mesh) == 0:
        raise IOError( 'No \'vtkUnstructuredGrid\' found!' )
    elif len(vtk_mesh) > 1:
        raise IOError( 'More than one \'vtkUnstructuredGrid\' found!' )

    # Cut off trailing '_' from array names.
    for k in xrange( vtk_mesh[0].GetPointData().GetNumberOfArrays() ):
        array = vtk_mesh[0].GetPointData().GetArray(k)
        array_name = array.GetName()
        if array_name[-1] == '_':
            array.SetName( array_name[0:-1] )

    #time_values = reader.GetOutputInformation(0).Get(vtkStreamingDemandDrivenPipeline.TIME_STEPS())

    return vtk_mesh[0] #, time_values
# ==============================================================================
def _read_points( vtk_mesh ):

    num_points = vtk_mesh.GetNumberOfPoints()

    # construct the points list
    points = np.empty( num_points, np.dtype((float, 3)))
    for k in range(num_points):
        points[k] = np.array( vtk_mesh.GetPoint( k ) )

    return points
# ==============================================================================
def _read_cells_nodes( vtk_mesh ):

    num_cells = vtk_mesh.GetNumberOfCells()
    # Assume that all cells have the same number of local nodes.
    max_num_local_nodes = vtk_mesh.GetCell(0).GetNumberOfPoints()
    cells_nodes = np.empty(num_cells, dtype = np.dtype((int, max_num_local_nodes)))

    for k in xrange(num_cells):
        cell = vtk_mesh.GetCell(k)
        num_local_nodes = cell.GetNumberOfPoints()
        assert num_local_nodes == max_num_local_nodes, 'Cells not uniform.'
        if num_local_nodes == max_num_local_nodes:
            # Gather up the points.
            for l in xrange( num_local_nodes ):
                cells_nodes[k][l] = cell.GetPointId( l )

    return cells_nodes
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
        array_name = array.GetName()
        num_entries = array.GetNumberOfTuples()
        num_components = array.GetNumberOfComponents()
        out[array_name] = np.empty((num_entries, num_components))
        for k in xrange( num_entries ):
            for i in xrange( num_components ):
                out[array_name][k][i] = array.GetComponent(k, i)

    return out
# ==============================================================================
def _read_field_data( vtk_data ):
    '''Gather field data.
    '''
    vtk_field_data = vtk_data.GetFieldData()
    num_arrays = vtk_field_data.GetNumberOfArrays()

    field_data = {}
    for k in xrange( num_arrays ):
        array  = vtk_field_data.GetArray(k)
        name   = array.GetName()
        num_values = array.GetDataSize()
        # Data type as specified in vtkSetGet.h.
        data_type = array.GetDataType()
        if data_type == 1:
            dtype = np.bool
        elif data_type in [2,3]:
            dtype = np.str
        elif data_type in [4, 5, 6, 7, 8, 9]:
            dtype = np.int
        elif data_type in [10, 11]:
            dtype = np.float
        else:
            raise TypeError('Unknown VTK data type %d.' % data_type)
        values = np.empty( num_values, dtype=dtype )
        for i in xrange( num_values ):
            values[i] = array.GetValue(i)
        field_data[ name ] = values

    return field_data
# ==============================================================================
