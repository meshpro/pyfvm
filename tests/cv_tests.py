# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of PyFVM.
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

import pyfvm


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
        vol = numpy.linalg.norm(mesh.control_volumes, ord=1)
        self.assertAlmostEqual(actual_values[0], vol, delta=tol)
        # Check control volume norms.
        norm = numpy.linalg.norm(mesh.control_volumes, ord=2)
        self.assertAlmostEqual(actual_values[1], norm, delta=tol)
        norm = numpy.linalg.norm(mesh.control_volumes, ord=numpy.Inf)
        self.assertAlmostEqual(actual_values[2], norm, delta=tol)
        return

    def test_degenerate_small(self):
        points = numpy.array([[0, 0], [1, 0], [0.5, 0.1], [0.5, -0.1]])
        # Manually compute the volumes.
        total_vol = 2 * 0.5 * 0.1
        cv0 = 0.25 * 0.1/0.5 * (0.5**2 + 0.1**2)
        cv = [cv0, cv0, 0.5*(total_vol-2*cv0), 0.5 * (total_vol-2*cv0)]
        #cells = numpy.array([[0, 1, 2], [0, 1, 3]])
        mesh = pyfvm.mesh2d.mesh2d(points, cells=None)
        actual_values = [numpy.linalg.norm(cv, ord=1),
                         numpy.linalg.norm(cv, ord=2),
                         numpy.linalg.norm(cv, ord=numpy.Inf)
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_rectanglesmall(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'rectanglesmall.e')
        mesh, _, _ = pyfvm.reader.read(filename)
        actual_values = [10.0,
                         5.0,
                         2.5]
        self._run_test(mesh, actual_values)
        return

    def test_arrow3d(self):
        nodes = numpy.array([[0.0,  0.0, 0.0],
                             [2.0, -1.0, 0.0],
                             [2.0,  0.0, 0.0],
                             [2.0,  1.0, 0.0],
                             [0.5,  0.0, -0.9],
                             [0.5,  0.0, 0.9]])
        cellsNodes = numpy.array([[1, 2, 4, 5],
                                  [2, 3, 4, 5],
                                  [0, 1, 4, 5],
                                  [0, 3, 4, 5]
                                  ])
        mesh = pyfvm.meshTetra.meshTetra(nodes, cellsNodes)
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
        mesh, _, _ = pyfvm.reader.read(filename)
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
        mesh, _, _ = pyfvm.reader.read(filename)
        actual_values = [302.52270072101,
                         15.3857579093391,
                         1.12779746704366
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_shell(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'shell.e')
        mesh, _, _ = pyfvm.reader.read(filename)
        actual_values = [3.46410161513775,
                         1.63299316185545,
                         1.15470053837925
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_sphere(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'sphere.e')
        mesh, _, _ = pyfvm.reader.read(filename)
        actual_values = [11.9741927059035,
                         1.39047542328083,
                         0.198927169088121
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_cubesmall(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'cubesmall.e')
        mesh, _, _ = pyfvm.reader.read(filename)
        actual_values = [10.0,
                         3.53553390593274,
                         1.25
                         ]
        self._run_test(mesh, actual_values)
        return

    def test_brick(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'brick-w-hole.e')
        mesh, _, _ = pyfvm.reader.read(filename)
        actual_values = [388.68629169464117,
                         16.661401941985677,
                         1.4684734547497671
                         ]
        self._run_test(mesh, actual_values)
        return

if __name__ == '__main__':
    unittest.main()
