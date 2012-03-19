#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Creates meshes on a 3D L-shape.
'''
import numpy as np
import time

import voropy
# ==============================================================================
def _main():

    # get the file name to be written to
    args = _parse_options()

    # circumcirlce radius
    cc_radius = 10.0
    lx = 2.0/np.sqrt(3.0) * cc_radius
    l = [lx, lx, lx]

    # create the mesh data structure
    print 'Create mesh...',
    start = time.time()
    # Corner points of the cube
    points = [( -0.5*l[0], -0.5*l[1], -0.5*l[2] ),
              (  0.5*l[0], -0.5*l[1], -0.5*l[2] ),
              (  0.5*l[0],  0.5*l[1], -0.5*l[2] ),
              ( -0.5*l[0],  0.5*l[1], -0.5*l[2] ),
              ( -0.5*l[0], -0.5*l[1],  0.5*l[2] ),
              (  0.5*l[0],  0.5*l[1],  0.5*l[2] ),
              ( -0.5*l[0],  0.5*l[1],  0.5*l[2] ),
              (  0.0,      -0.5*l[1],  0.5*l[2] ),
              (  0.0,      -0.5*l[1],  0.0 ),
              (  0.5*l[0], -0.5*l[1],  0.0 ),
              (  0.5*l[0],  0.0,       0.0 ),
              (  0.5*l[0],  0.0,       0.5*l[2] ),
              (  0.0,       0.0,       0.5*l[2] ),
              (  0.0,       0.0,       0.0 )
              ]
    facets = [[0,1,2,3],
              [4,7,12,11,5,6],
              [0,1,9,8,7,4],
              [1,2,5,11,10,9],
              [2,5,6,3],
              [3,6,4,0],
              [8,13,12,7],
              [8,9,10,13],
              [10,11,12,13]
              ]
    # create the mesh
    import meshpy.tet
    # Set the geometry and build the mesh.
    info = meshpy.tet.MeshInfo()
    info.set_points( points )
    info.set_facets( facets )
    meshpy_mesh = meshpy.tet.build( info,
                                    max_volume = args.maxvol
                                  )
    mesh = voropy.mesh3d(np.array(meshpy_mesh.points),
                         meshpy_mesh.elements)
    elapsed = time.time() - start
    print 'done. (%gs)' % elapsed

    num_nodes = len(mesh.node_coords)
    print '\n%d nodes, %d elements\n' % (num_nodes, len(mesh.cells))

    print 'Create values...',
    start = time.time()
    # create values
    X = np.empty( num_nodes, dtype = complex )
    for k, node in enumerate(mesh.node_coords):
        X[k] = complex( 1.0, 0.0 )
        #X[k] = complex( sin( x/lx * np.pi ), sin( y/ly * np.pi ) )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    print 'Create thickness values...',
    start = time.time()
    # add thickness value
    thickness = np.empty( num_nodes, dtype = float )
    alpha = 1.0
    beta = 2.0
    for k, node in enumerate(mesh.node_coords):
        #thickness[k] = alpha + (beta-alpha) * (y/(0.5*ly))**2
        thickness[k] = 1.0
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    print 'Create mvp...',
    start = time.time()
    # add magnetic vector potential
    A = np.empty( (num_nodes,3), dtype = float )
    # exact corner of a cube
    phi = np.pi/4.0 # azimuth
    theta = np.arctan( 1.0/np.sqrt(2.0) ) # altitude
    height0 = 0.1
    height1 = 1.1
    radius = 2.0
    import magnetic_vector_potentials
    for k, node in enumerate(mesh.node_coords):
        A[k,:] = magnetic_vector_potentials.mvp_z( node )
        #A[k,:] = magnetic_vector_potentials.mvp_magnetic_dot( node, radius, height0, height1 )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # write the mesh with data
    print 'Write to file...',
    start = time.time()
    mesh.write(args.filename, {'psi': X, 'A': A})
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct a trival tetrahedrization of a 3D L-shape.' )


    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'file to be written to'
                       )

    parser.add_argument( '--maxvol', '-m',
                         metavar = 'MAXVOL',
                         dest='maxvol',
                         nargs='?',
                         type=float,
                         const=1.0,
                         default=1.0,
                         help='maximum tetrahedron volume of the tetrahedrization (default: 1.0)'
                       )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
