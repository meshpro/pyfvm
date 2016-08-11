# -*- coding: utf-8 -*-
import meshzoo
import pyamg
import pyfvm
from pyfvm.form_language import *
import numpy


class EnergyEdgeKernel(object):
    def __init__(self):
        self.subdomains = ['everywhere']
        return

    def eval(self, mesh, edge_ids):
        X = mesh.node_coords[mesh.edges['nodes'][edge_ids]]
        x0 = X[:, 0, :].T
        x1 = X[:, 1, :].T
        edge_midpoint = 0.5 * (x0 + x1)
        edge = x1 - x0
        edge_ce_ratio = mesh.ce_ratios[edge_ids]

        beta = 1.0

        return numpy.array([
            [edge_ce_ratio, -edge_ce_ratio * numpy.exp(1j * beta)],
            [-edge_ce_ratio * numpy.exp(-1j * beta), edge_ce_ratio]
            ])

vertices, cells = meshzoo.rectangle.create_mesh(0.0, 2.0, 0.0, 1.0, 101, 51)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

matrix = pyfvm.get_fvm_matrix(mesh, [EnergyEdgeKernel()], [], [], [])
rhs = mesh.control_volumes.copy()

# Smoothed aggregation.
sa = pyamg.smoothed_aggregation_solver(matrix, smooth='energy')
u = sa.solve(rhs, tol=1e-10)

# Cannot write complex data ot VTU; split real and imaginary parts first.
u2 = numpy.ascontiguousarray(numpy.vstack((u.real, u.imag)).T)
mesh.write('out.vtu', point_data={'u': u2})
