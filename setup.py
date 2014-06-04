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
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='voropy',
      version='0.1.2',
      author='Nico Schlömer',
      author_email='nico.schloemer@gmail.com',
      packages=['voropy'],
      description='Delaunay meshes, Voronoi regions',
      long_description=read('README.md'),
      url='https://github.com/nschloe/VoroPy',
      download_url=None,
      license='License :: OSI Approved :: BSD License',
      platforms='any',
      requires=['numpy', 'scipy', 'vtk'],
      classifiers=['Development Status :: 3 - Alpha',
                   'Topic :: Utilities'
                   ],
      scripts=['examples/ball',
               'examples/convert_mesh',
               'examples/cube',
               'examples/cylinder_tri',
               'examples/delaunay_checker',
               'examples/delaunay_maker',
               'examples/ellipse',
               'examples/hexagon',
               'examples/lshape',
               'examples/lshape3d',
               'examples/moebius2_tri',
               'examples/moebius_tri',
               'examples/moebius_tri_alt',
               'examples/pacman',
               'examples/pseudomoebius',
               'examples/rectangle',
               'examples/rectangle_with_hole',
               'examples/simple_arrow',
               'examples/sphere',
               'examples/tetrahedron',
               'examples/triangle',
               'examples/tube'],
      )
# Don't install test files.
#        data_files=[('tests', ['tests/cubesmall.e',
#                               'tests/pacman.e',
#                               'tests/rectanglesmall.e',
#                               'tests/test.e',
#                               'tests/tetrahedron.e'])]
