# TODO create poisson.py by calling `form-compiler def.py poisson.py`
import singular

import meshzoo
import pyfvm
from scipy.sparse import linalg

# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle.create_mesh(1.0, 1.0, 51, 51, zigzag=True)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

problem = singular.Singular(mesh)

x = linalg.spsolve(problem.matrix, problem.rhs)

mesh.write('out.vtu', point_data={'x': x})
