class ${name}(object):
    def __init__(self, mesh):
        self.mesh = mesh
        ${members_init}
        return

    def eval(self, k):
        ${vertex_body}
        return (
            ${vertex_contrib},
            ${vertex_affine}
            )
