# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of VoroPy.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from this
#  software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
'''
Module for reading unstructured grids (and related data) from various file
formats.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import os
import meshio
import numpy
import voropy

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
            return voropy.mesh2d.mesh2d(points[:, :2], cells_nodes), \
                point_data, field_data
        else:  # 2d shell mesh
            return voropy.meshTri.meshTri(points, cells_nodes), \
                   point_data, field_data
    elif len(cells_nodes[0]) == 4:  # 3D
        return voropy.meshTetra.meshTetra(points, cells_nodes), \
               point_data, field_data
    else:
        raise RuntimeError('Unknown mesh type.')
    return
