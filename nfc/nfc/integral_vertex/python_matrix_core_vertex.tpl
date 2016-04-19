class ${name}(object):
  def __init__(self):
      ${members_init}
      return

  def eval(self, k):
      ${vertex_body}
      return (
          ${vertex_contrib},
          ${vertex_affine}
          )
