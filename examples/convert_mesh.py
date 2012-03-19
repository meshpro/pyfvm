#! /usr/bin/env python
'''
Convert a mesh file to another.
'''
import voropy
import numpy as np
# ==============================================================================
def _main():
    # Parse command line arguments.
    args = _parse_options()

    # read mesh data
    mesh, point_data, field_data = voropy.read(args.in_filename, timestep=args.timestep)

    # write it out
    mesh.write(args.out_filename, point_data, field_data)

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Convert mesh formats.' )

    parser.add_argument( 'in_filename',
                         metavar = 'INFILE',
                         type    = str,
                         help    = 'mesh file to be read from'
                       )

    parser.add_argument( 'out_filename',
                         metavar = 'OUTFILE',
                         type    = str,
                         help    = 'mesh file to be written to'
                       )

    parser.add_argument( '--timestep', '-t',
                         metavar='TIMESTEP',
                         dest='timestep',
                         nargs='?',
                         type=int,
                         const=0,
                         default=0,
                         help='read a particular time step (default: 0)'
                       )

    return parser.parse_args()
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
