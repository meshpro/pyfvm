#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Creates a mesh on a rectangle in the x-y-plane.
'''
import numpy as np
import time

import magnetic_vector_potentials
import voropy
# ==============================================================================
def _main():

    # get the file name to be written to
    args = _parse_options()

    # dimensions of the rectangle
    cc_radius = 15.0 # circumcircle radius
    lx = np.sqrt(2.0) * cc_radius
    l = [lx, lx]

    # create the mesh data structure
    print 'Create mesh...',
    start = time.time()
    if args.cnx != 0:
        mymesh = _canonical(l, [args.cnx, args.cnx])
    elif args.znx != 0:
        mymesh = _zigzag(l, [args.znx, args.znx])
    elif args.maxarea != 0.0:
        mymesh = _meshpy(l, args.maxarea)
    else:
        raise RuntimeError('Set either -c or -z or -m.')
    elapsed = time.time() - start
    print 'done. (%gs)' % elapsed

    num_nodes = len(mymesh.nodes)
    print '\n%d nodes, %d elements\n' % (num_nodes, len(mymesh.cellsNodes))

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
    A = np.empty(num_nodes, dtype = np.dtype((float,3)))
    height0 = 0.1
    height1 = 1.1
    radius = 2.0
    for k, node in enumerate(mymesh.nodes):
        A[k] = magnetic_vector_potentials.mvp_z( node )
        #A[k] = magnetic_vector_potentials.mvp_magnetic_dot( node, radius, height0, height1 )
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
def _canonical(l, N):
    # Create the vertices.
    x_range = np.linspace( -0.5*l[0], 0.5*l[0], N[0] )
    y_range = np.linspace( -0.5*l[1], 0.5*l[1], N[1] )
    num_nodes = len(x_range) * len(y_range)
    nodes = np.empty(num_nodes, dtype=np.dtype((float, 2)))
    k = 0
    for x in x_range:
        for y in y_range:
            nodes[k] = np.array([x, y])
            k += 1

    # create the elements (cells)
    num_elems = 2 * (N[0]-1) * (N[1]-1)
    elems = np.empty(num_elems, dtype=np.dtype((int,3)))
    k = 0
    for i in xrange(N[0] - 1):
        for j in xrange(N[1] - 1):
            elems[k] = np.array([i*N[1] + j, (i + 1)*N[1] + j + 1,  i     *N[1] + j + 1])
            k += 1
            elems[k] = np.array([i*N[1] + j, (i + 1)*N[1] + j    , (i + 1)*N[1] + j + 1])
            k += 1

    return voropy.mesh2d(nodes, elems)
# ==============================================================================
def _zigzag(l, N):
    # Create the vertices.
    x_range = np.linspace(-0.5*l[0], 0.5*l[0], N[0])
    y_range = np.linspace(-0.5*l[1], 0.5*l[1], N[1])

    num_nodes = len(x_range) * len(y_range)
    nodes = np.empty(num_nodes, dtype=np.dtype((float, 2)))
    k = 0
    for x in x_range:
        for y in y_range:
            nodes[k] = np.array([x, y])
            k += 1

    # create the elements (cells)
    num_elems = 2 * (N[0]-1) * (N[1]-1)
    elems = np.empty(num_elems, dtype=np.dtype((int,3)))
    k = 0
    for i in xrange(N[0] - 1):
        for j in xrange(N[1] - 1):
            if (i+j)%2 == 0:
                elems[k] = np.array([i*N[1] + j, (i + 1)*N[1] + j + 1,  i     *N[1] + j + 1])
                k += 1
                elems[k] = np.array([i*N[1] + j, (i + 1)*N[1] + j    , (i + 1)*N[1] + j + 1])
                k += 1
            else:
                elems[k] = np.array([i    *N[1] + j, (i+1)*N[1] + j  , i*N[1] + j+1])
                k += 1
                elems[k] = np.array([(i+1)*N[1] + j, (i+1)*N[1] + j+1, i*N[1] + j+1])
                k += 1

    return voropy.mesh2d(nodes, elems)
# ==============================================================================
def _meshpy(l, max_area):
    import mesh.meshpy_interface

    # corner points
    points = [ ( -0.5*l[0], -0.5*l[1] ),
               (  0.5*l[0], -0.5*l[1] ),
               (  0.5*l[0],  0.5*l[1] ),
               ( -0.5*l[0],  0.5*l[1] ) ]

    return mesh.meshpy_interface.create_mesh(max_area, points)
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct a triangulation of a rectangle.' )


    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'file to be written to'
                       )

    parser.add_argument( '--canonical', '-c',
                         metavar = 'NX',
                         dest='cnx',
                         nargs='?',
                         type=int,
                         const=0,
                         default=0,
                         help='canonical triangulation with NX discretization points along each axis'
                       )

    parser.add_argument( '--zigzag', '-z',
                         metavar = 'NX',
                         dest='znx',
                         nargs='?',
                         type=int,
                         const=0,
                         default=0,
                         help='zigzag triangulation with NX discretization points along each axis'
                       )

    parser.add_argument( '--meshpy', '-m',
                         metavar = 'MAXAREA',
                         dest='maxarea',
                         nargs='?',
                         type=float,
                         const=0.0,
                         default=0.0,
                         help='meshpy triangulation with MAXAREA maximum triangle area'
                       )

    return parser.parse_args()
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
