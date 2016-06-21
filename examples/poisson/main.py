import pyfvm
pyfvm.compiler.compile('def.py', 'poisson.py')

import poisson

import meshzoo
from scipy.sparse import linalg

# Read the mesh using meshio
# mesh, _, _ = pyfvm.reader.read('pacman.e')

# Create mesh using meshzoo
vertices, cells = meshzoo.rectangle.create_mesh(2.0, 1.0, 21, 11, zigzag=True)
mesh = pyfvm.meshTri.meshTri(vertices, cells)

mesh.mark_subdomains([
        poisson.Gamma0(),
        poisson.Gamma1()
        ])

problem = poisson.Poisson(mesh)

x = linalg.spsolve(problem.matrix, problem.rhs)

mesh.write('out.vtu', point_data={'x': x})
