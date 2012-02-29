# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (C) 2012 Nico Schlömer
#
# This file is part of VoroPy.
#
# VoroPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# VoroPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with VoroPy.  If not, see <http://www.gnu.org/licenses/>.
#
# ==============================================================================
from distutils.core import setup

setup( name='voropy',
       version='0.0.1',
       packages=['voropy'],
       url = 'https://bitbucket.org/nschloe/voropy',
       download_url='https://bitbucket.org/nschloe/voropy/downloads',
       author = 'Nico Schl"omer',
       author_email = 'nico.schloemer@gmail.com',
       description = 'Delaunay meshes, Voronoi regions',
       license = 'GNU Lesser General Public License (LGPL), Version 3',
       platforms='any',
       requires=['numpy','scipy','vtk']
     )
