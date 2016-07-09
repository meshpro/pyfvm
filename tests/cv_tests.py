# -*- coding: utf-8 -*-
#
import os
import numpy
import unittest

import pyfvm


class TestControlVolumes(unittest.TestCase):

    def setUp(self):
        return

    def _run_test(self, mesh, volume, cv_norms, covol_norms):
        tol = 1.0e-5

        if mesh.cells['nodes'].shape[1] == 3:
            dim = 2
        elif mesh.cells['nodes'].shape[1] == 4:
            dim = 3
        else:
            raise ValueError('Can only handle triangles and tets.')

        # Check the volume by summing over the cell volume.
        vol2 = sum(mesh.cell_volumes)

        self.assertAlmostEqual(volume, vol2, delta=tol)

        # Check the volume by summing over the absolute value of the
        # control volumes.
        vol = numpy.sum(mesh.control_volumes)
        self.assertAlmostEqual(volume, vol, delta=tol)

        # Check control volume norms.
        norm = numpy.linalg.norm(mesh.control_volumes, ord=2)
        self.assertAlmostEqual(cv_norms[0], norm, delta=tol)
        norm = numpy.linalg.norm(mesh.control_volumes, ord=numpy.Inf)
        self.assertAlmostEqual(cv_norms[1], norm, delta=tol)

        # Check the volume by summing over the
        #   1/n * edge_lengths * covolumes
        # covolumes.
        total_covolume = numpy.sum(mesh.edge_lengths * mesh.covolumes / dim)
        self.assertAlmostEqual(volume, total_covolume, delta=tol)

        # Check covolume norms.
        norm = numpy.linalg.norm(mesh.covolumes, ord=2)
        self.assertAlmostEqual(covol_norms[0], norm, delta=tol)
        norm = numpy.linalg.norm(mesh.covolumes, ord=numpy.Inf)
        self.assertAlmostEqual(covol_norms[1], norm, delta=tol)

        return

    def test_degenerate_small0(self):
        points = numpy.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0.01, 0.0],
            ])
        cells = numpy.array([[0, 1, 2]])
        mesh = pyfvm.meshTri.meshTri(points, cells)
        self._run_test(
                mesh,
                0.005,
                [3.8268185015427627, 3.12625],
                [21.650635675842587, 12.50249975229104]
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
                [0.60207972893961459, 0.325],
                [3.5014282800022278, 2.4]
                )
        return

    def test_rectanglesmall(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'rectanglesmall.e')
        mesh, _, _ = pyfvm.reader.read(filename)
        self._run_test(
                mesh,
                10,
                [5.0, 2.5],
                [7.1063352028243898, 5.0]
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
                [0.54867112189361633, 0.354],
                [4.6093865583659497, 2.4709512338368973]
                )
        return

    def test_tetrahedron(self):
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'tetrahedron.e'
            )
        mesh, _, _ = pyfvm.reader.read(filename)
        # mesh.show_edge(54)
        self._run_test(
                mesh,
                64.150028545707983,
                [15.633459930030972, 9.0023269417919636],
                [22.456543028439334, 12.09471520942393]
                )
        return

    def test_pacman(self):
        filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'pacman.e'
                )
        mesh, _, _ = pyfvm.reader.read(filename)
        self._run_test(
                mesh,
                302.52270072101,
                [15.3857579093391, 1.12779746704366],
                [21.636574419194687, 1.3500278827154624]
                )
        return

    def test_shell(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'shell.e')
        mesh, _, _ = pyfvm.reader.read(filename)
        self._run_test(
                mesh,
                3.46410161513775,
                [1.63299316185545, 1.15470053837925],
                [1.8257417943354759, 0.81649655229931284]
                )
        return

    def test_sphere(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'sphere.e')
        mesh, _, _ = pyfvm.reader.read(filename)
        self._run_test(
                mesh,
                11.9741927059035,
                [1.39047542328083, 0.198927169088121],
                [5.1108705055302739, 0.60864468986577691]
                )
        return

    def test_cubesmall(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'cubesmall.e')
        mesh, _, _ = pyfvm.reader.read(filename)
        self._run_test(
                mesh,
                10.0,
                [3.7267798925256708, 5.0/3.0],
                [5.7759558765734713, 2.3452374507983533]
                )
        return

    def test_brick(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'brick-w-hole.e')
        mesh, _, _ = pyfvm.reader.read(filename)
        self._run_test(
                mesh,
                388.68629169464117,
                [16.885287218950758, 1.532783118899316],
                [28.247403902696792, 1.9147280888306519]
                )
        return

if __name__ == '__main__':
    unittest.main()
