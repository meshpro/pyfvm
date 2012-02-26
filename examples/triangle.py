#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time

import voropy
# ==============================================================================
def _main():

    # get the command line arguments
    args = _parse_options()

    #Circumcircle radius of the triangle.
    cc_radius = 5.0

    # Create initial nodes/elements.
    #nodes = [ cc_radius * np.array([np.cos((0.5+0.0/3.0)*pi), np.sin((0.5+0.0/3.0)*pi), 0]),
              #cc_radius * np.array([np.cos((0.5+2.0/3.0)*pi), np.sin((0.5+2.0/3.0)*pi), 0]),
              #cc_radius * np.array([np.cos((0.5+4.0/3.0)*pi), np.sin((0.5+4.0/3.0)*pi), 0])
            #]
    nodes = cc_radius * np.array([np.array([ 0.0, 1.0]),
                                  np.array([-0.5*np.sqrt(3.0), -0.5]),
                                  np.array([ 0.5*np.sqrt(3.0), -0.5])])
    cells = np.array([[0, 1, 2]], dtype=int)

    # Create mesh data structure.
    mesh = voropy.mesh2d(nodes, cells)
    mesh.create_adjacent_entities()

    # Refine..
    print 'Mesh refinement...',
    start = time.time()
    for k in xrange(args.ref_steps):
        mesh.refine()
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    num_nodes = len( mesh.nodes )

    # create values
    print 'Create values...',
    start = time.time()
    X = np.empty( num_nodes, dtype = complex )
    for k, node in enumerate(mesh.nodes):
        #import random, cmath
        #X[k] = cmath.rect( random.random(), 2.0 * pi * random.random() )
        X[k] = complex( 1.0, 0.0 )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # Add magnetic vector potential.
    print 'Create mvp...',
    start = time.time()
    A = np.empty( (num_nodes,3), dtype = float )
    import magnetic_vector_potentials
    height0 = 0.1
    height1 = 1.1
    radius = 2.0
    for k, node in enumerate(mesh.nodes):
        A[k,:] = magnetic_vector_potentials.mvp_z( node )
        #A[k,:] = magnetic_vector_potentials.mvp_magnetic_dot( node, radius, height0, height1 )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # write the mesh
    print 'Write mesh...',
    start = time.time()
    mesh.write(args.filename, {'psi':X, 'A':A})
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    print '\n%d nodes, %d elements' % (num_nodes, len(mesh.cellsNodes))

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct triangulation of a triangle.' )

    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'file to be written to'
                       )

    parser.add_argument( '--refinements', '-r',
                         metavar='NUM_REFINEMENTS',
                         dest='ref_steps',
                         nargs='?',
                         type=int,
                         const=0,
                         default=0,
                         help='number of mesh refinement steps to be performed (default: 0)'
                       )


    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
