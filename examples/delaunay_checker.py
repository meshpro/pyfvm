#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
import voropy
import numpy as np
# ==============================================================================
def _main():
    '''Main function.
    '''
    filename = _parse_input_arguments()

    # read the mesh
    print 'Reading the mesh...',
    mesh, _, _ = voropy.read( filename )
    print 'done.'

    if mesh.is_delaunay():
        print 'The given mesh is a Delaunay mesh.'
    else:
        print 'The given mesh is NOT a Delaunay mesh.'

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

    args = parser.parse_args()

    return args.filename
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
