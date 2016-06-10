class ${name}(object):
    def __init__(self, mesh):
        self.mesh = mesh

    def eval(self, k):
        ${eval_body}
        return ([[
            ${edge00},
            ${edge01}
            ],
            [
            ${edge10},
            ${edge11}
            ]],
            [${edge_affine0}, ${edge_affine1}]
            )
