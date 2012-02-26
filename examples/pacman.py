#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Creates a mesh for a circle with a cut.
'''
import numpy as np
import time

from mesh import mesh, meshpy_interface, magnetic_vector_potentials
# ==============================================================================
def _main():

    args = _parse_options()

    n_phi = args.num_boundary_points
    radius = 5.0

    # set those to 0.0 for perfect circle
    cut_angle = 0.1 * 2*np.pi
    cut_deepness = 0.5 * radius

    # Choose the maximum area of a triangle equal to the area of
    # an equilateral triangle on the boundary.
    a_boundary = (2*np.pi-cut_angle)*radius / n_phi
    max_area = a_boundary**2 * np.sqrt(3.0) / 4.0
    max_area = float( max_area ) # meshpy can't deal with numpy.float64

    # generate points on the boundary
    Phi = np.linspace( 0.5*cut_angle,
                       2*np.pi - 0.5*cut_angle,
                       n_phi,
                       endpoint = False
                     )
    points = []
    if abs(cut_angle)>0.0 or cut_deepness!=0.0:
        points.append( (radius-cut_deepness, 0.0) )
    for phi in Phi:
        points.append((radius * np.cos(phi), radius * np.sin(phi)))

    # create the mesh
    print 'Meshpy...',
    start = time.time()
    mymesh = meshpy_interface.create_mesh( max_area, points )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    num_nodes = len( mymesh.nodes )

    # create values
    X = np.empty( num_nodes, dtype = complex )
    for k, x in enumerate( mymesh.nodes ):
        X[k] = complex( 1.0, 0.0 )

    # add thickness value
    thickness = np.empty( num_nodes, dtype = float )
    for k, x in enumerate( mymesh.nodes ):
        thickness[k] = 1.0

    # Add magnetic vector potential.
    print 'Create mvp...',
    start = time.time()
    A = np.empty(num_nodes, dtype=np.dtype((float,3)))
    height0 = 0.1
    height1 = 1.1
    radius = 2.0
    for k, node in enumerate(mymesh.nodes):
        A[k] = magnetic_vector_potentials.mvp_z( node )
        #A[k,:] = magnetic_vector_potentials.mvp_magnetic_dot( node, radius, height0, height1 )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # write the mesh
    print 'Write mesh...',
    start = time.time()
    mymesh.write( args.filename, {'psi':X, 'A':A, 'thickness':thickness} )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    print '\n%d nodes, %d elements' % (len(mymesh.nodes), len(mymesh.cellsNodes))

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct a MeshPy triangulation of a circle with a cut.' )

    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'file to be written to'
                       )

    parser.add_argument( '--num-boundary-points', '-b',
                         metavar='NUM_BOUNDARY_POINTS',
                         dest='num_boundary_points',
                         nargs='?',
                         type=int,
                         const=10,
                         default=10,
                         help='number of points on the outer boundary (default: 10)'
                       )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
