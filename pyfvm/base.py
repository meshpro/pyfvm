# -*- coding: utf-8 -*-
#
import meshio
import numpy

__all__ = []


class _base_mesh(object):

    def __init__(self,
                 nodes,
                 cells_nodes
                 ):
        self.node_coords = nodes
        return

    def write(self,
              filename,
              point_data=None,
              cell_data=None,
              field_data=None
              ):
        if self.node_coords.shape[1] == 2:
            n = len(self.node_coords)
            a = numpy.ascontiguousarray(
                numpy.c_[self.node_coords, numpy.zeros(n)]
                )
        else:
            a = self.node_coords

        if self.cells['nodes'].shape[1] == 3:
            cell_type = 'triangle'
        elif self.cells['nodes'].shape[1] == 4:
            cell_type = 'tetrahedron'
        else:
            raise RuntimeError('Only triangles/tetrahedra supported')

        meshio.write(
            filename,
            a,
            {cell_type: self.cells['nodes']},
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
