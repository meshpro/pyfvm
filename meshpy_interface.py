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
        return mesh2d(mesh.points, mesh.elements)
    else:
        mesh = create_triangle_mesh( roundtrip_points, holes )
        return mesh3d(mesh.points, mesh.elements)

    return
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
