class ${name}:
  public nosh::matrix_core_vertex
{
  public:
    ${name}(const std::shared_ptr<const nosh::mesh> & mesh)${members_init} {}

    virtual ~${name}() {}

    virtual
    nosh::vertex_data
    eval(const moab::EntityHandle & vertex) const
    {
      ${vertex_body}
      return {
        ${vertex_contrib},
        ${vertex_affine}
        };
    }

  private:
    ${members_declare}
}; // class ${name}
