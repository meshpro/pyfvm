import numpy as np
import unittest

import voropy
# ==============================================================================
class TestMesh(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, mesh, actual_values ):
        # Compute the control volumes.
        if mesh.control_volumes is None:
            mesh.compute_control_volumes()

        tol = 1.0e-12

        # Check the volume by summing over the cell volume.
        if mesh.cellsVolume is None:
            mesh.create_cells_volume()
        vol2 = sum(mesh.cellsVolume)
        self.assertAlmostEqual( actual_values[0], vol2, delta=tol )

        # Check the volume by summing over the absolute value of the
        # control volumes.
        vol = np.linalg.norm( mesh.control_volumes, ord=1 )
        self.assertAlmostEqual( actual_values[0], vol, delta=tol )

        # Check control volume norms.
        norm = np.linalg.norm( mesh.control_volumes, ord=2 )
        self.assertAlmostEqual( actual_values[1], norm, delta=tol )
        norm = np.linalg.norm( mesh.control_volumes, ord=np.Inf)
        self.assertAlmostEqual( actual_values[2], norm, delta=tol )

        return
    # --------------------------------------------------------------------------
    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        mesh, _, _ = voropy.read( filename )
        actual_values = [ 10.0,
                          5.0,
                          2.5 ]

        self._run_test(mesh, actual_values)
        return
    # --------------------------------------------------------------------------
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
        mesh = voropy.mesh3d(nodes, cellsNodes)

        # pull this to see what a negative covolume looks like
        #if mesh.edgesNodes is None:
            #mesh.create_adjacent_entities()
        #mesh.show_edge(5)

        actual_values = [ 1.2,
                          0.58276428453480855,
                          0.459 ]
        self._run_test(mesh, actual_values)
        return
    # --------------------------------------------------------------------------
    def test_tetrahedron(self):
        filename = 'tetrahedron.e'
        mesh, _, _ = voropy.read( filename )

        #mesh.show_edge(54)

        actual_values = [ 64.150028545708011,
                          15.243602636687179,
                          7.7180603065060023 ]
        self._run_test(mesh, actual_values)
        return
    # --------------------------------------------------------------------------
    def test_pacman(self):
        filename = 'pacman.e'
        mesh, _, _ = voropy.read( filename )
        actual_values = [ 302.52270072101,
                          15.3857579093391,
                          1.12779746704366 ]

        self._run_test(mesh, actual_values)
        return
    # --------------------------------------------------------------------------
    def test_cubesmall(self):
        filename = 'cubesmall.e'
        mesh, _, _ = voropy.read( filename )
        actual_values = [ 10.0,
                          3.53553390593274,
                          1.25 ]
        self._run_test(mesh, actual_values)
        return
    # --------------------------------------------------------------------------
    def test_brick(self):
        filename = 'brick-w-hole.e'
        mesh, _, _ = voropy.read( filename )

        actual_values = [ 388.68629169464117,
                          16.661401941985677,
                          1.4684734547497671 ]
        self._run_test(mesh, actual_values)
        return
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
