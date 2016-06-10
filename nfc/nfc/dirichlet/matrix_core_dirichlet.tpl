class ${name}:
  public nosh::matrix_core_dirichlet
{
  public:
    ${name}(const std::shared_ptr<const nosh::mesh> & mesh): ${init} {}

    virtual ~${name}() {}

    virtual double
    eval(const moab::EntityHandle & vertex) const {
      ${eval_body}return ${eval_return_value};
    }
}; // class ${name}
