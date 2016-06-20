# -*- coding: utf-8 -*-
#
from . import compiler
from . import linear_fvm_problem
from . import mesh2d
from . import meshTri
from . import meshTetra
from . import reader

__all__ = [
    'linear_fvm_problem',
    'mesh2d',
    'meshTri',
    'meshTetra',
    'reader'
    ]
__version__ = '0.1.0'
__author__ = 'Nico Schlömer'
__author_email__ = 'nico.schloemer@gmail.com'
