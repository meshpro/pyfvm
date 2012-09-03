#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Creates a simplistic triangular mesh the sphere.
'''
import vtk
import mesh, mesh_io
import numpy as np
from math import pi, sin, cos
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Mesh parameters
    n_phi   = 100 # number of points on each circle of latitude (except poles)
    n_theta = 50 # number of theta-levels, including poles
    radius  = 1.0
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Generate suitable ranges for parametrization
    phi_range = np.linspace( 0.0, 2*pi, num = n_phi, endpoint = False )
    theta_range = np.linspace( -pi/2  + pi/(n_theta-1), pi/2 - pi/(n_theta-1), num = n_theta-2 )

    # Create the vertices.
    nodes = []

    # south pole
    south_pole_index = 0
    nodes.append( mesh.Node( [ 0.0, 0.0, -radius ] ) )

    # nodes in the circles of latitude (except poles)
    for theta in theta_range:
        for phi in phi_range:
            nodes.append( mesh.Node( [ radius * cos(theta) * sin(phi),
                                       radius * cos(theta) * cos(phi),
                                       radius * sin(theta)
                                     ]
                                   )
                       )
    # north pole
    north_pole_index = len( nodes )
    nodes.append( mesh.Node( [ 0.0, 0.0, radius ] ) )

    ## create the elements (cells)
    elems = []
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # connections to south pole
    for i in xrange(n_phi - 1):
        elem_nodes = [ south_pole_index, i+1, i+2 ]
        elems.append( mesh.Cell( elem_nodes, [], vtk.VTK_TRIANGLE ) )
    # close geometry
    elem_nodes = [ south_pole_index, n_phi, 1]
    elems.append( mesh.Cell( elem_nodes, [], vtk.VTK_TRIANGLE ) )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # non-pole elements
    for i in xrange(n_theta - 3):
        for j in xrange(n_phi - 1):
            elem_nodes = [ i*n_phi + j+1, i*n_phi + j+2, (i+1)*n_phi + j+2 ]
            elems.append( mesh.Cell( elem_nodes, [], vtk.VTK_TRIANGLE ) )

            elem_nodes = [ i*n_phi + j+1, (i+1)*n_phi + j+2, (i+1)*n_phi + j + 1 ]
            elems.append( mesh.Cell( elem_nodes, [], vtk.VTK_TRIANGLE ) )

    # close the geometry
    for i in xrange(n_theta - 3):
        elem_nodes = [ (i+1)*n_phi, i*n_phi + 1, (i+1)*n_phi + 1 ]
        elems.append( mesh.Cell( elem_nodes, [], vtk.VTK_TRIANGLE ) )
        elem_nodes = [ (i+1)*n_phi, (i+1)*n_phi + 1, (i+2)*n_phi ]
        elems.append( mesh.Cell( elem_nodes, [], vtk.VTK_TRIANGLE ) )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # connections to the north pole
    for i in range(n_phi - 1):
        elem_nodes = [ i+1 + n_phi*(n_theta-3) + 1,
                       i   + n_phi*(n_theta-3) + 1,
                       north_pole_index
                     ]
        elems.append( mesh.Cell( elem_nodes, [], vtk.VTK_TRIANGLE ) )
    # close geometry
    elem_nodes = [ 0       + n_phi*(n_theta-3) + 1,
                   n_phi-1 + n_phi*(n_theta-3) + 1,
                   north_pole_index
                 ]
    elems.append( mesh.Cell( elem_nodes, [], vtk.VTK_TRIANGLE ) )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # add values
    num_nodes = len( nodes )
    X = np.empty( num_nodes, dtype = complex )
    k = 0
    # south pole
    X[k] = complex( 1.0, 0.0 )
    k += 1
    for phi in phi_range:
        for theta in theta_range:
            X[k] = complex( 1.0, 0.0 )
            k += 1
    # north pole
    X[k] = complex( 1.0, 0.0 )
    k += 1

    # add parameters
    params = { "mu": 0.0 }

    # create the mesh data structure
    mymesh = mesh.Mesh( nodes, elems )

    # create the mesh
    mesh_io.write_mesh( "sphere.e",
                        mymesh,
                        [X], ["psi"],
                        params
                      )
# ------------------------------------------------------------------------------
