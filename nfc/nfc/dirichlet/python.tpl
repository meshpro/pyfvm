class ${name}(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.subdomains = ${init_subdomains}
        return

    def eval(self, k):
        return ${eval_return_value}
