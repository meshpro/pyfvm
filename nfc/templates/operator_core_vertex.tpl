class ${name}:
  public nosh::operator_core_vertex
{
  public:
    ${name}(const std::shared_ptr<const nosh::mesh> & mesh)${members_init} {}

    virtual ~${name}() {}

    virtual
      double
      eval(
        const moab::EntityHandle & vertex,
        const Teuchos::ArrayRCP<const double> & u
        ) const
      {
        ${eval_body}
        return ${return_value};
      }

    ${methods}

  private:
    ${members_declare}
}; // class ${name}
