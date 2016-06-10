class ${name}:
  public nosh::matrix_core_edge
{
  public:
    ${name}(const std::shared_ptr<const nosh::mesh> & mesh)${members_init} {}

    virtual ~${name}() {}

    virtual
      nosh::matrix_core_edge_data
      eval(const moab::EntityHandle & edge) const
      {
        ${eval_body}
        return {
          {
            {
              ${edge00},
                ${edge01},
            },
              {
                ${edge10},
                ${edge11},
              }
          },
          {
            ${edge_affine0},
            ${edge_affine1}
          }
        };
      }

  private:
    ${members_declare}
}; // class ${name}
