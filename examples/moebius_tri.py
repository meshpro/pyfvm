#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Creates a simplistic triangular mesh on a M\"obius strip.
'''
import vtk
import mesh, mesh_io
import numpy as np
from math import pi, sin, cos
# ==============================================================================
def _main():

    # get the file name to be written to
    file_name = _parse_options()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Mesh parameters
    # Number of nodes along the length of the strip
    nl = 190
    # Number of nodes along the width of the strip (>= 2)
    nw = 31
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # The width of the strip
    width = 1.0
    scale = 10.0

    # radius of the strip when flattened out
    r = 1.0

    #l = 5
    p = 1.5

    # seam displacement
    alpha0 = 0.0 # pi / 2

    # How flat the strip will be.
    # Positive values result in left-turning M\"obius strips, negative in
    # right-turning ones.
    # Also influences the width of the strip
    flatness = 1.0

    # How many twists are there in the "paper"?
    moebius_index = 1
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Generate suitable ranges for parametrization
    u_range = np.linspace( 0.0, 2*pi, num = nl, endpoint = False )
    v_range = np.linspace( -0.5*width, 0.5*width, num = nw )

    # Create the vertices. This is based on the parameterization
    # of the M\"obius strip as given in
    # <http://en.wikipedia.org/wiki/M%C3%B6bius_strip#Geometry_and_topology>
    nodes = []
    for u in u_range:
        pre_alpha = 0.5 * u
        #if u > pi:
            #pre_alpha = pi / 2 * abs( u/pi -1 )**l + pi / 2
        #elif u < pi:
            #pre_alpha = - pi / 2 * abs( u/pi -1 )**l + pi / 2
        #else:
            #pre_alpha = pi / 2
        #if u > pi:
            #pre_alpha = pi / 2 * ( 1 - (1-abs(u/pi-1)**p)**(1/p) ) + pi / 2
        #elif u < pi:
            #pre_alpha = - pi / 2 * ( 1 - (1-abs(u/pi-1)**p)**(1/p) ) + pi / 2
        #else:
            #pre_alpha = pi / 2
        alpha = moebius_index * pre_alpha + alpha0
        for v in v_range:
            nodes.append( mesh.Node( [ scale * ( r + v*cos(alpha) ) * cos(u),
                                       scale * ( r + v*cos(alpha) ) * sin(u),
                                       flatness * scale * v*sin(alpha)
                                     ]
                                   )
                       )

    # create the elements (cells)
    elems = []
    for i in range(nl - 1):
        for j in range(nw - 1):
            elem_nodes = [ i*nw + j, (i + 1)*nw + j + 1,  i     *nw + j + 1 ]
            elems.append( mesh.Cell( elem_nodes) )
            elem_nodes = [ i*nw + j, (i + 1)*nw + j    , (i + 1)*nw + j + 1 ]
            elems.append( mesh.Cell( elem_nodes) )
    # close the geometry
    if moebius_index % 2 == 0:
        # Close the geometry upside up (even M\"obius fold)
        for j in range(nw - 1):
            elem_nodes = [ (nl - 1)*nw + j, j + 1 , (nl - 1)*nw + j + 1 ]
            elems.append( mesh.Cell( elem_nodes) )
            elem_nodes = [ (nl - 1)*nw + j, j     , j + 1  ]
            elems.append( mesh.Cell( elem_nodes) )
    else:
        # Close the geometry upside down (odd M\"obius fold)
        for j in range(nw - 1):
            elem_nodes = [ (nl-1)*nw + j, (nw-1) - (j+1) , (nl-1)*nw +  j+1  ]
            elems.append( mesh.Cell( elem_nodes) )
            elem_nodes = [ (nl-1)*nw + j, (nw-1) - j     , (nw-1)    - (j+1) ]
            elems.append( mesh.Cell( elem_nodes) )


    # create the mesh data structure
    mymesh = mesh.Mesh( nodes, elems )

    # create the mesh
    mymesh.write(file_name)

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import optparse, sys

    usage = "usage: %prog outfile"

    parser = optparse.OptionParser( usage = usage )

    (options, args) = parser.parse_args()

    if not args  or  len(args) != 1:
        parser.print_help()
        sys.exit( "\nProvide a file to be written to." )

    return args[0]
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
