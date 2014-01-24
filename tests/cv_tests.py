import os
import numpy as np
import unittest

import voropy


class TestControlVolumes(unittest.TestCase):

    def setUp(self):
        return

    def _run_test(self, mesh, actual_values):
        # Compute the control volumes.
        if mesh.control_volumes is None:
            mesh.compute_control_volumes()
        tol = 1.0e-12
        # Check the volume by summing over the cell volume.
        if mesh.cell_volumes is None:
            mesh.create_cell_volumes()
        vol2 = sum(mesh.cell_volumes)
        self.assertAlmostEqual(actual_values[0], vol2, delta=tol)
        # Check the volume by summing over the absolute value of the
        # control volumes.
        vol = np.linalg.norm(mesh.control_volumes, ord=1)
        self.assertAlmostEqual(actual_values[0], vol, delta=tol)
        # Check control volume norms.
        norm = np.linalg.norm(mesh.control_volumes, ord=2)
        self.assertAlmostEqual(actual_values[1], norm, delta=tol)
        norm = np.linalg.norm(mesh.control_volumes, ord=np.Inf)
        self.assertAlmostEqual(actual_values[2], norm, delta=tol)
        return

    def test_degenerate_small(self):
        points = np.array([[0, 0], [1, 0], [0.5, 0.1], [0.5, -0.1]])
        # Manually compute the volumes.
        total_vol = 2 * 0.5 * 0.1
        cv0 = 0.25 * 0.1/0.5 * (0.5**2 + 0.1**2)
        cv = [cv0, cv0, 0.5*(total_vol-2*cv0), 0.5 * (total_vol-2*cv0)]
        #cells = np.array([[0, 1, 2], [0, 1, 3]])
        mesh = voropy.mesh2d(points, cells=None)
        actual_values = [np.linalg.norm(cv, ord=1),
                         np.linalg.norm(cv, ord=2),
                         np.linalg.norm(cv, ord=np.Inf)
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_rectanglesmall(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'rectanglesmall.e')
        mesh, _, _ = voropy.read(filename)
        actual_values = [10.0,
                         5.0,
                         2.5]
        self._run_test(mesh, actual_values)
        return

    def test_arrow3d(self):
        nodes = np.array([[0.0,  0.0, 0.0],
                          [2.0, -1.0, 0.0],
                          [2.0,  0.0, 0.0],
                          [2.0,  1.0, 0.0],
                          [0.5,  0.0, -0.9],
                          [0.5,  0.0, 0.9]])
        cellsNodes = np.array([[1, 2, 4, 5],
                               [2, 3, 4, 5],
                               [0, 1, 4, 5],
                               [0, 3, 4, 5]])
        mesh = voropy.meshTetra(nodes, cellsNodes)
        # pull this to see what a negative covolume looks like
        #if mesh.edgesNodes is None:
            #mesh.create_adjacent_entities()
        #mesh.show_edge(5)
        actual_values = [1.2,
                         0.58276428453480855,
                         0.459
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_tetrahedron(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'tetrahedron.e')
        mesh, _, _ = voropy.read(filename)
        #mesh.show_edge(54)
        actual_values = [64.150028545708011,
                         15.243602636687179,
                         7.7180603065060023
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_pacman(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'pacman.e')
        mesh, _, _ = voropy.read(filename)
        actual_values = [302.52270072101,
                         15.3857579093391,
                         1.12779746704366
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_shell(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'shell.e')
        mesh, _, _ = voropy.read(filename)
        actual_values = [3.46410161513775,
                         1.63299316185545,
                         1.15470053837925
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_sphere(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'sphere.e')
        mesh, _, _ = voropy.read(filename)
        actual_values = [11.9741927059035,
                         1.39047542328083,
                         0.198927169088121
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_cubesmall(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'cubesmall.e')
        mesh, _, _ = voropy.read(filename)
        actual_values = [10.0,
                         3.53553390593274,
                         1.25
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_brick(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'brick-w-hole.e')
        mesh, _, _ = voropy.read(filename)
        actual_values = [388.68629169464117,
                         16.661401941985677,
                         1.4684734547497671
                         ]
        self._run_test(mesh, actual_values)
        return

if __name__ == '__main__':
    unittest.main()
