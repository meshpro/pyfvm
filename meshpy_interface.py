'''Interface to MeshPy.
'''
# ==============================================================================
import numpy as np
# ==============================================================================
MAX_AREA = 0.0
# ==============================================================================
def create_mesh( max_area, roundtrip_points, facets = None, holes=None ):
    '''Create a mesh.
    '''
    import meshpy.triangle

    global MAX_AREA
    MAX_AREA = max_area

    if facets is not None:
        mesh = create_tetrahedron_mesh( roundtrip_points, facets )
    else:
        mesh = create_triangle_mesh( roundtrip_points, holes )

    return _construct_mymesh( mesh )
# ==============================================================================
def create_triangle_mesh(roundtrip_points, holes):
    '''Create a mesh.
    '''
    import meshpy.triangle

    # Set the geometry and build the mesh.
    info = meshpy.triangle.MeshInfo()
    info.set_points( roundtrip_points )
    if holes is not None:
        info.set_holes(holes)
    info.set_facets( _round_trip_connect(0, len(roundtrip_points)-1) )

    meshpy_mesh = meshpy.triangle.build(info,
                                        refinement_func = _needs_refinement
                                        )

    return meshpy_mesh
# ==============================================================================
def create_tetrahedron_mesh( roundtrip_points, facets ):
    '''Create a mesh.
    '''
    import meshpy.tet

    # Set the geometry and build the mesh.
    info = meshpy.tet.MeshInfo()
    print info
    info.set_points( roundtrip_points )
    info.set_facets( facets )

    meshpy_mesh = meshpy.tet.build( info,
                                    max_volume = MAX_AREA
                                  )

    return meshpy_mesh
# ==============================================================================
def _round_trip_connect(start, end):
    '''Return pairs of subsequent numbers from start to end.
    '''
    result = []
    for i in range(start, end):
        result.append((i, i+1))
    result.append((end, start))
    return result
# ==============================================================================
def _needs_refinement( vertices, area ):
    '''Refinement function.'''
    return area > MAX_AREA
# ==============================================================================
def _construct_mymesh( meshpy_mesh ):
    '''Create the mesh entity.'''
    import mesh
    import vtk

    # Create the vertices.
    num_nodes = len(meshpy_mesh.points)
    nodes = np.empty(num_nodes, dtype=np.dtype((float, 3)))
    for k, point in enumerate(meshpy_mesh.points):
        if len(point) == 2:
            nodes[k][:2] = point
            nodes[k][2] = 0.0
        elif len(point) == 3:
            nodes[k] = point
        else:
            raise ValueError('Unknown point.')
          
    # Create the elements (cells).
    num_elems = len(meshpy_mesh.elements)
    # Take the dimension of the first cell to be the dimension of all cells.
    dim_elems = len(meshpy_mesh.elements[0])
    elems = np.empty(num_elems, dtype=np.dtype((int,dim_elems)))
    for k, element in enumerate(meshpy_mesh.elements):
        elems[k] = element
    
    # create the mesh data structure
    return mesh.Mesh( nodes, elems )
# ==============================================================================
