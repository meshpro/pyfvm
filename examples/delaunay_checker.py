#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
import voropy
import numpy as np
# ==============================================================================
def _main():
    '''Main function.
    '''
    args = _parse_input_arguments()

    # read the mesh
    print 'Reading the mesh...',
    mesh, _, _ = voropy.read( args.filename )
    print 'done.'

    num_delaunay_violations, num_interior_edges = mesh.check_delaunay()

    if num_delaunay_violations == 0:
        print 'The given mesh is a Delaunay mesh.'
    else:
        alpha = float(num_delaunay_violations) / num_interior_edges
        print 'Delaunay condition NOT fulfilled on %d of %d interior edges/faces (%g%%).' \
            % (num_delaunay_violations, num_interior_edges, alpha*100)

    return
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser(description = 'Check if a mesh is a Delaunay mesh.')

    parser.add_argument('filename',
                        metavar = 'FILE',
                        type    = str,
                        help    = 'ExodusII file containing the geometry'
                        )

    return parser.parse_args()
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
