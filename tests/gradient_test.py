# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of VoroPy.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from this
#  software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
import os
import numpy
import unittest

import voropy


class GradientTest(unittest.TestCase):

    def setUp(self):
        return

    def _run_test(self, mesh):
        num_nodes = len(mesh.node_coords)
        # Create function  2*x + 3*y.
        a_x = 7.0
        a_y = 3.0
        a0 = 1.0
        u = a_x * mesh.node_coords[:, 0] \
            + a_y * mesh.node_coords[:, 1] \
            + a0 * numpy.ones(num_nodes)
        # Get the gradient analytically.
        sol = numpy.empty((num_nodes, 2))
        sol[:, 0] = a_x
        sol[:, 1] = a_y
        # Compute the gradient numerically.
        grad_u = mesh.compute_gradient(u)
        mesh.write('test.e', point_data={'diff': grad_u-sol})
        tol = 1.0e-5
        for k in range(num_nodes):
            self.assertAlmostEqual(grad_u[k][0], sol[k][0], delta=tol)
            self.assertAlmostEqual(grad_u[k][1], sol[k][1], delta=tol)
        return

    def test_pacman(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'pacman.e')
        mesh, _, _ = voropy.reader.read(filename)
        self._run_test(mesh)
        return

if __name__ == '__main__':
    unittest.main()
