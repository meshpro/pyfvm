# -*- coding: utf-8 -*-
#
import os
from math import fsum
import numpy
import unittest

import pyfvm


class TestVolumes(unittest.TestCase):

    def setUp(self):
        return

    def _run_test(self, mesh, volume, cv_norms, covol_norms, cellvol_norms):
        tol = 1.0e-12

        if mesh.cells['nodes'].shape[1] == 3:
            dim = 2
        elif mesh.cells['nodes'].shape[1] == 4:
            dim = 3
        else:
            raise ValueError('Can only handle triangles and tets.')

        # Check cell volumes.
        total_cellvolume = fsum(mesh.cell_volumes)
        self.assertAlmostEqual(volume, total_cellvolume, delta=tol * volume)
        norm = numpy.linalg.norm(mesh.cell_volumes, ord=2)
        self.assertAlmostEqual(cellvol_norms[0], norm, delta=tol)
        norm = numpy.linalg.norm(mesh.cell_volumes, ord=numpy.Inf)
        self.assertAlmostEqual(cellvol_norms[1], norm, delta=tol)

        # Check the volume by summing over the
        #   1/n * edge_lengths * covolumes
        # covolumes.
        total_covolume = fsum(mesh.edge_lengths * mesh.covolumes / dim)
        self.assertAlmostEqual(volume, total_covolume, delta=tol * volume)
        # Check covolume norms.
        alpha = fsum(mesh.covolumes**2)
        self.assertAlmostEqual(covol_norms[0], alpha, delta=tol)
        alpha = max(abs(mesh.covolumes))
        self.assertAlmostEqual(covol_norms[1], alpha, delta=tol)

        # Check the volume by summing over the absolute value of the
        # control volumes.
        vol = fsum(mesh.control_volumes)
        self.assertAlmostEqual(volume, vol, delta=tol * volume)
        # Check control volume norms.
        norm = numpy.linalg.norm(mesh.control_volumes, ord=2)
        self.assertAlmostEqual(cv_norms[0], norm, delta=tol)
        norm = numpy.linalg.norm(mesh.control_volumes, ord=numpy.Inf)
        self.assertAlmostEqual(cv_norms[1], norm, delta=tol)

        return

    def test_degenerate_small0(self):
        points = numpy.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1.0e-2, 0.0],
            ])
        cells = numpy.array([[0, 1, 2]])
        mesh = pyfvm.meshTri.meshTri(points, cells)
        self._run_test(
                mesh,
                0.005,
                [3.8268185015427632, 3.12625],
                [468.750025, numpy.sqrt(156.3125)],
                [0.005, 0.005]
                )
        return

    def test_degenerate_small1(self):
        points = numpy.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0.1, 0.0],
            [0.5, -0.1, 0.0]
            ])
        cells = numpy.array([[0, 1, 2], [0, 1, 3]])
        # Manually compute the volumes.
        total_vol = 2 * 0.5 * 0.1
        mesh = pyfvm.meshTri.meshTri(points, cells)
        self._run_test(
                mesh,
                total_vol,
                [numpy.sqrt(0.3625), 0.325],
                [12.26, 2.4],
                [numpy.sqrt(1.0/200.0), 0.05]
                )
        return

    def test_degenerate_tet0(self):
        h = 1.0e-1
        points = numpy.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.25, 0.25, h],
            ])
        cells = numpy.array([[0, 1, 2, 3]])
        # Manually compute the volumes.
        total_vol = 1.0/3.0 * 0.5 * h
        mesh = pyfvm.meshTetra.meshTetra(points, cells)
        self._run_test(
                mesh,
                total_vol,
                [0.12038850913902652, 77.0/720.0],
                [11791.0/28800.0, numpy.sqrt(127.0/1152.0)],
                [1.0/60.0, 1.0/60.0]
                )
        return

    def test_degenerate_tet1(self):
        h = 1.0e-1
        points = numpy.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.25, 0.25, h],
            [0.25, 0.25, -h],
            ])
        cells = numpy.array([
            [0, 1, 2, 3],
            [0, 1, 2, 4]
            ])
        # Manually compute the volumes.
        total_vol = 2 * 1.0/3.0 * 0.5 * h
        mesh = pyfvm.meshTetra.meshTetra(points, cells)
        self._run_test(
                mesh,
                total_vol,
                [0.18734818957173291, 77.0/720.0],
                [1211.0/1200.0, 23.0/60.0],
                [1.0 / numpy.sqrt(2.0) / 30., 1.0/60.0]
                )
        return

    def test_rectanglesmall(self):
        points = numpy.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
            ])
        cells = numpy.array([
            [0, 1, 2],
            [0, 2, 3]
            ])
        mesh = pyfvm.meshTri.meshTri(points, cells)
        self._run_test(
                mesh,
                10,
                [5.0, 2.5],
                [50.5, 5.0],
                [numpy.sqrt(50.0), 5.0]
                )
        return

    def test_arrow3d(self):
        nodes = numpy.array([
            [0.0,  0.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0,  0.0, 0.0],
            [2.0,  1.0, 0.0],
            [0.5,  0.0, -0.9],
            [0.5,  0.0, 0.9]
            ])
        cellsNodes = numpy.array([
            [1, 2, 4, 5],
            [2, 3, 4, 5],
            [0, 1, 4, 5],
            [0, 3, 4, 5]
            ])
        mesh = pyfvm.meshTetra.meshTetra(nodes, cellsNodes)
        # pull this to see what a negative covolume looks like
        # mesh.show_edge(5)
        self._run_test(
                mesh,
                1.2,
                [numpy.sqrt(0.30104), 0.354],
                [95609. / 4500., numpy.sqrt(6.1056)],
                [numpy.sqrt(0.45), 0.45]
                )
        return

    def test_tetrahedron(self):
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'tetrahedron.vtu'
            )
        mesh, _, _ = pyfvm.reader.read(filename)
        # mesh.show_edge(54)
        self._run_test(
                mesh,
                64.1500299099584,
                [17.07120343309435, 7.5899731568813653],
                [227.9618079370525, 4.5503630826356547],
                [11.571692332290635, 2.9699087921277054]
                )
        return

    def test_pacman(self):
        filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'pacman.vtu'
                )
        mesh, _, _ = pyfvm.reader.read(filename)
        self._run_test(
                mesh,
                73.64573933105898,
                [3.596101914906618, 0.26638548094154707],
                [115.99321112012936, 0.67825038377950408],
                [2.6213234038171014, 0.13841739494523228]
                )
        return

    def test_shell(self):
        points = numpy.array([
            [0.0,  0.0,  1.0],
            [1.0,  0.0,  0.0],
            [0.0,  1.0,  0.0],
            [-1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0]
            ])
        cells = numpy.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 1, 4]
            ])
        mesh = pyfvm.meshTri.meshTri(points, cells)
        self._run_test(
                mesh,
                2 * numpy.sqrt(3),
                [2 * numpy.sqrt(2.0/3.0), 2.0/numpy.sqrt(3.0)],
                [10.0 / 3.0, numpy.sqrt(2.0 / 3.0)],
                [numpy.sqrt(3.0), numpy.sqrt(3.0) / 2.0]
                )
        return

    def test_sphere(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'sphere.vtu')
        mesh, _, _ = pyfvm.reader.read(filename)
        self._run_test(
                mesh,
                12.273645818711595,
                [1.0177358705967492, 0.10419690304323895],
                [27.574311895740863, 0.34998394343357359],
                [0.72653362732751214, 0.05350373815413411]
                )
        return

    def test_cubesmall(self):
        points = numpy.array([
            [-0.5, -0.5, -5.0],
            [-0.5,  0.5, -5.0],
            [0.5, -0.5, -5.0],
            [-0.5, -0.5,  5.0],
            [0.5,  0.5, -5.0],
            [0.5,  0.5,  5.0],
            [-0.5,  0.5,  5.0],
            [0.5, -0.5,  5.0]
            ])
        cells = numpy.array([
            [0, 1, 2, 3],
            [1, 2, 4, 5],
            [1, 2, 3, 5],
            [1, 3, 5, 6],
            [2, 3, 5, 7]
            ])
        mesh = pyfvm.meshTetra.meshTetra(points, cells)
        self._run_test(
                mesh,
                10.0,
                [numpy.sqrt(5.0) * 5.0/3.0, 5.0/3.0],
                [20017.0/600.0, numpy.sqrt(39601.0 / 7200.0)],
                [numpy.sqrt(2.0) * 10.0/3.0, 10.0/3.0]
                )
        return

    def test_toy(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'toy.vtu')
        mesh, _, _ = pyfvm.reader.read(filename)
        self._run_test(
                mesh,
                9.3875504672601107,
                [0.20348466631551548, 0.010271101930468585],
                [14.649583739489584, 0.81666108124852155],
                [0.091903119589148916, 0.0019959463063558944]
                )
        return

if __name__ == '__main__':
    unittest.main()
