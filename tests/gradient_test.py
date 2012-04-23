import numpy as np
import unittest

import voropy
# ==============================================================================
class GradientTest(unittest.TestCase):
      # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, mesh):

        tol = 1.0e-12

        num_nodes = len(mesh.node_coords)
        # Create function  2*x + 3*y.
        a_x = 2.0
        a_y = 3.0
        a0 = 4.0
        u = np.array(a_x*mesh.node_coords[:,0] + a_y*mesh.node_coords[:,1] + a0)

        # Get the gradient analytically.
        sol = np.empty((num_nodes,2))
        sol[:,0] = a_x
        sol[:,1] = a_y

        # Compute the gradient numerically.
        grad_u = mesh.compute_gradient(u)

        for k in xrange(num_nodes):
            self.assertAlmostEqual( grad_u[k][0], sol[k][0], delta=1.0e-6 )
            self.assertAlmostEqual( grad_u[k][1], sol[k][1], delta=1.0e-6 )

        return
    # --------------------------------------------------------------------------
    def test_pacman(self):
        filename = 'pacman.e'
        mesh, _, _ = voropy.read( filename )

        self._run_test(mesh)
        return
    # --------------------------------------------------------------------------
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
