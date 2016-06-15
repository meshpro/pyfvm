# TODO create poisson.py by calling `form-compiler def.py poisson.py`
import poisson

import pyfvm
from scipy.sparse import linalg

# Read the mesh
mesh, _, _ = pyfvm.reader.read('pacman.e')

mesh.mark_subdomains([
        poisson.Gamma0(),
        poisson.Gamma1()
        ])

problem = poisson.Poisson(mesh)

x = linalg.spsolve(problem.matrix, problem.rhs)

mesh.write('out.vtu', point_data={'x': x})
