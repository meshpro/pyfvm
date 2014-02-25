from . import mesh2d
from . import meshTri
from . import meshTetra
from . import reader

__version__ = '0.0.2'

__all__ = filter(lambda s: not s.startswith('_'), dir())
