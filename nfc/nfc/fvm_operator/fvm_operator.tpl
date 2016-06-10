class ${name}:
  public nosh::fvm_operator
{
  public:
    ${name}(
        ${constructor_args}
        ):
      nosh::fvm_operator(
        _mesh,
        ${init_edge_cores},
        ${init_vertex_cores},
        ${init_boundary_cores},
        ${init_dirichlets},
        ${init_operators}
        )
      ${members_init}
    {
    }

    virtual
      ~${name}()
      {}

    ${extra_methods}

  private:
    ${members_declare}
}; // class ${name}
