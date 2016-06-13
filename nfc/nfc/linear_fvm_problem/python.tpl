class ${name}(pyfvm.linear_fvm_problem.LinearFvmProblem):
    def __init__(self, mesh):
        self.mesh = mesh
        ${members_init}
        super(${name}, self).__init__(
          mesh=mesh,
          edge_cores=self.edge_cores,
          vertex_cores=self.vertex_cores,
          boundary_cores=self.boundary_cores,
          dirichlets=self.dirichlets
          )
        return
