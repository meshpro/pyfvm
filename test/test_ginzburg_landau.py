# -*- coding: utf-8 -*-
#
"""
These tests are against the reference values for pynosh.
"""
import os

import pyfvm

import meshplex
import numpy
import pytest


class Energy(object):
    """Specification of the kinetic energy operator.
    """

    def __init__(self, mu):
        self.magnetic_field = mu * numpy.array([0.0, 0.0, 1.0])
        self.subdomains = [None]

        return

    def eval(self, mesh, cell_mask):
        """This eval does as if the magnetic vector potential was defined only in the
        nodes, and interpolates from there. It'd also be possible to simply evaluate it
        at the edge midpoints, but we do the former here to do the same as pynosh.
        """
        nec = mesh.idx_hierarchy[..., cell_mask]
        X = mesh.node_coords[nec]

        magnetic_potential = numpy.array([
            0.5 * numpy.cross(self.magnetic_field, x)
            for x in mesh.node_coords
        ])

        edge = X[1] - X[0]
        edge_ce_ratio = mesh.ce_ratios[..., cell_mask]

        # The dot product <magnetic_potential, edge>, executed for many
        # points at once; cf. <http://stackoverflow.com/a/26168677/353337>.
        # beta = numpy.einsum("ijk,ijk->ij", magnetic_potential.T, edge.T)
        mp_edge = 0.5 * (magnetic_potential[nec[0]] + magnetic_potential[nec[1]])
        beta = numpy.einsum("...k,...k->...", mp_edge, edge)

        return numpy.array(
            [
                [edge_ce_ratio, -edge_ce_ratio * numpy.exp(1j * beta)],
                [-edge_ce_ratio * numpy.exp(-1j * beta), edge_ce_ratio],
            ]
        )


@pytest.mark.parametrize(
    "filename, control_values",
    [
        ("rectanglesmall.e", [0.0063121712308067401, 10.224658806561596]),
        ("pacman.e", [0.37044264296585938, 10.000520856079092]),
        # geometric ce_ratios
        ("cubesmall.e", [0.00012499993489764605, 10.062484361987309]),
        ("brick-w-hole.e", [0.167357712543159, 12.05581968511059]),
        # # Algebraic ce_ratios:
        # ("cubesmall.e", [8.3541623155714007e-05, 10.058364522531498]),
        # ("brick-w-hole.e", [0.16763276012920181, 15.131119904340618]),
    ],
)
def test(filename, control_values):
    this_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(this_path, filename)
    mu = 1.0e-2

    # read the mesh
    mesh, point_data, field_data, _ = meshplex.read(filename)

    keo = pyfvm.get_fvm_matrix(mesh, edge_kernels=[Energy(mu)])

    tol = 1.0e-13

    # Check that the matrix is Hermitian.
    KK = keo - keo.H
    assert abs(KK.sum()) < tol

    # Check the matrix sum.
    assert abs(control_values[0] - keo.sum()) < tol

    # Check the 1-norm of the matrix |Re(K)| + |Im(K)|.
    # This equals the 1-norm of the matrix defined by the block
    # structure
    #   Re(K) -Im(K)
    #   Im(K)  Re(K).
    K = abs(keo.real) + abs(keo.imag)
    assert abs(control_values[1] - numpy.max(K.sum(0))) < tol
    return
