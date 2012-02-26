#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Creates a mesh on a rectangle in the x-y-plane.
'''
from mesh import mesh, magnetic_vector_potentials, meshpy_interface
import numpy as np
import time
# ==============================================================================
def _main():

    # get the file name to be written to
    args = _parse_options()

    # dimensions of the rectangle
    cc_radius = 15.0 # circumcircle radius
    lx = np.sqrt(2.0) * cc_radius
    l = [lx, lx]

    h_radius = 1.0

    # create the mesh data structure
    print 'Create mesh...',
    start = time.time()
    # corner points
    points = [( 0.5*l[0],  0.0 ),
              ( 0.5*l[0],  0.5*l[1]),
              (-0.5*l[0],  0.5*l[1]),
              (-0.5*l[0], -0.5*l[1]),
              ( 0.5*l[0], -0.5*l[1]),
              ( 0.5*l[0],  0.0 )
              ]
    # create circular boundary on the inside
    segments = 100
    for k in xrange(segments+1):
        angle = k * 2.0 * np.pi / segments
        points.append((h_radius * np.cos(angle), h_radius * np.sin(angle)))
    # mark the hole by an interior point
    holes = [(0,0)]
    #holes = None
    mymesh = meshpy_interface.create_mesh(args.maxarea, points, holes=holes)
    elapsed = time.time() - start
    print 'done. (%gs)' % elapsed

    num_nodes = len(mymesh.nodes)
    print '\n%d nodes, %d elements\n' % (num_nodes, len(mymesh.cells))

    # create values
    print 'Create values...',
    start = time.time()
    X = np.empty( num_nodes, dtype = complex )
    for k, node in enumerate(mymesh.nodes):
        #X[k] = cmath.rect( random.random(), 2.0 * pi * random.random() )
        X[k] = complex( 1.0, 0.0 )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # Add magnetic vector potential.
    print 'Create mvp...',
    start = time.time()
    A = np.empty( (num_nodes,3), dtype = float )
    height0 = 0.1
    height1 = 1.1
    radius = 2.0
    for k, node in enumerate(mymesh.nodes):
        A[k,:] = magnetic_vector_potentials.mvp_z( node )
        #A[k,:] = magnetic_vector_potentials.mvp_magnetic_dot( node, radius, height0, height1 )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # write the mesh
    print 'Write mesh...',
    start = time.time()
    mymesh.write( args.filename, {'psi':X, 'A':A} )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct a triangulation of a rectangle with a circular hole.' )


    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'file to be written to'
                       )

    parser.add_argument( '--maxarea', '-m',
                         metavar = 'MAXAREA',
                         dest='maxarea',
                         nargs='?',
                         type=float,
                         const=1.0,
                         default=1.0,
                         help='maximum triangle area (default: 1.0)'
                       )

    return parser.parse_args()
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
