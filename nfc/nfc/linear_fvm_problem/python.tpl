class ${name}(pyfvm.linear_fvm_problem.LinearFvmProblem):
    def __init__(self, mesh):
        self.mesh = mesh
        ${members_init}
        super(${name}, self).__init__()
        return
