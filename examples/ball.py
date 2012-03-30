#!/usr/bin/env python
# ==============================================================================
import argparse
import numpy as np
import time
from meshpy.tet import MeshInfo, build
from meshpy.geometry import generate_surface_of_revolution, EXT_OPEN, GeometryBuilder

import voropy
#import mesh
#import mesh.meshpy_interface
#import mesh.magnetic_vector_potentials
# ==============================================================================
def _main():

    args = _parse_options()

    radius = 5.0
    points = args.p

    radial_subdiv = 2 * points

    dphi = np.pi / points

    # Make sure the nodes meet at the poles of the ball.
    def truncate(r):
        if abs(r) < 1e-10:
            return 0
        else:
            return r

    # Build outline for surface of revolution.
    rz = [(truncate(radius * np.sin(i*dphi)), radius * np.cos(i*dphi))
          for i in xrange(points+1)
         ]

    print 'Build mesh...',
    start = time.time()
    geob = GeometryBuilder()
    geob.add_geometry( *generate_surface_of_revolution(rz,
                                                       closure=EXT_OPEN,
                                                       radial_subdiv=radial_subdiv)
                     )
    mesh_info = MeshInfo()
    geob.set(mesh_info)
    meshpy_mesh = build(mesh_info)
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # Fill the data into a voropy mesh object.
    mesh = voropy.mesh3d(meshpy_mesh.points, meshpy_mesh.elements)

    print '\n%d nodes, %d elements' % (len(mesh.nodes), len(mesh.cellsNodes))

    # write the mesh
    print 'Write mesh...',
    start = time.time()
    mesh.write(args.filename)
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct tetrahedrization of a ball.' )


    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'file to be written to'
                       )

    parser.add_argument( '--numpoints', '-p',
                         metavar = 'N',
                         dest='p',
                         nargs='?',
                         type=int,
                         const=10,
                         default=10,
                         help    = 'number of discretization points along a logitudinal line'
                       )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
