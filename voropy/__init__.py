from mesh2d import *
from meshTri import *
from meshTetra import *
from reader import *

__version__ = '0.0.2'

__all__ = filter(lambda s:not s.startswith('_'), dir())
