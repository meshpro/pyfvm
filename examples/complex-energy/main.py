# -*- coding: utf-8 -*-
import meshzoo
import pyamg
import pyfvm
from pyfvm.form_language import *
import numpy


class EnergyEdgeKernel(object):
    def __init__(self):
        mu = 0.1
        self.magnetic_field = mu * numpy.array([0.0, 0.0, 1.0])
        self.subdomains = ['everywhere']
        return

    def eval(self, mesh, edge_ids):
        X = mesh.node_coords[mesh.edges['nodes'][edge_ids]]
        x0 = X[:, 0, :].T
        x1 = X[:, 1, :].T
        edge_midpoint = 0.5 * (x0 + x1)
        edge = x1 - x0
        edge_ce_ratio = mesh.ce_ratios[edge_ids]

        # project the magnetic potential on the edge at the midpoint
        magnetic_potential = \
            0.5 * numpy.cross(self.magnetic_field, edge_midpoint.T).T

        # The dot product <magnetic_potential, edge>, executed for many points
        # at once; cf. <http://stackoverflow.com/a/26168677/353337>.
        beta = numpy.einsum('ij, ij->i', magnetic_potential.T, edge.T)

        return numpy.array([
            [edge_ce_ratio, -edge_ce_ratio * numpy.exp(1j * beta)],
            [-edge_ce_ratio * numpy.exp(-1j * beta), edge_ce_ratio]
            ])

vertices, cells = meshzoo.rectangle.create_mesh(0.0, 2.0, 0.0, 1.0, 101, 51)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

matrix = pyfvm.FvmMatrix(mesh, [EnergyEdgeKernel()], [], [], [])
rhs = mesh.control_volumes.copy()

# Smoothed aggregation.
sa = pyamg.smoothed_aggregation_solver(matrix.matrix, smooth='energy')
u = sa.solve(rhs, tol=1e-10)

# Cannot write complex data ot VTU; split real and imaginary parts first.
u2 = numpy.ascontiguousarray(numpy.vstack((u.real, u.imag)).T)
mesh.write('out.vtu', point_data={'u': u2})
