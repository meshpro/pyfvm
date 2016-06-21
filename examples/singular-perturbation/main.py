import pyfvm
pyfvm.compiler.compile('def.py', 'singular.py')

import singular

import meshzoo
from scipy.sparse import linalg

# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle.create_mesh(1.0, 1.0, 51, 51, zigzag=True)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

problem = singular.Singular(mesh)

x = linalg.spsolve(problem.matrix, problem.rhs)

mesh.write('out.vtu', point_data={'x': x})
