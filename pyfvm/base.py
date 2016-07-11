# -*- coding: utf-8 -*-
#
import meshio
import numpy

__all__ = []


def _row_dot(a, b):
    # http://stackoverflow.com/a/26168677/353337
    return numpy.einsum('ij, ij->i', a, b)


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
            cell_type = 'tetra'
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

    def compute_edge_lengths(self):
        edges = self.node_coords[self.edges['nodes'][:, 1]] \
            - self.node_coords[self.edges['nodes'][:, 0]]
        self.edge_lengths = numpy.sqrt(_row_dot(edges, edges))
        return
