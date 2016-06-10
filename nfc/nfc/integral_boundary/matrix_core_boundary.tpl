class ${name}: public nosh::matrix_core_boundary
{
  public:
    ${name}(const std::shared_ptr<const nosh::mesh> & mesh)${members_init} {}

    virtual ~${name}() {}

    virtual
      nosh::boundary_data
      eval(const moab::EntityHandle & vertex) const
      {
        ${body}
        return {
          ${coeff},
          ${affine}
          };
      }

  private:
    ${members_declare}
}; // class ${name}
