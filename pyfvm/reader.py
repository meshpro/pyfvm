# -*- coding: utf-8 -*-
#
'''
Module for reading unstructured grids (and related data) from various file
formats.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import os
import meshio
import numpy
import pyfvm

__all__ = ['read']


def read(filename, timestep=None):
    '''Reads an unstructured mesh with added data.

    :param filenames: The files to read from.
    :type filenames: str
    :param timestep: Time step to read from, in case of an Exodus input mesh.
    :type timestep: int, optional
    :returns mesh{2,3}d: The mesh data.
    :returns point_data: Point data read from file.
    :type point_data: dict
    :returns field_data: Field data read from file.
    :type field_data: dict
    '''
    points, cells_nodes, point_data, cell_data, field_data = \
        meshio.read(filename)

    if len(cells_nodes[0]) == 3:
        if all(points[:, 2] == 0.0):
            # Flat mesh.
            # Check if there's three-dimensional point data that can be cut.
            # Don't use iteritems() here as we want to be able to
            # set the value in the loop.
            for key, value in point_data.items():
                if len(value.shape) > 1 and \
                        value.shape[1] == 3 and \
                        all(value[:, 2] == 0.0):
                    point_data[key] = value[:, :2]
            return pyfvm.mesh2d.mesh2d(points[:, :2], cells_nodes), \
                point_data, field_data
        else:  # 2d shell mesh
            return pyfvm.meshTri.meshTri(points, cells_nodes), \
                   point_data, field_data
    elif len(cells_nodes[0]) == 4:  # 3D
        return pyfvm.meshTetra.meshTetra(points, cells_nodes), \
               point_data, field_data
    else:
        raise RuntimeError('Unknown mesh type.')
    return
