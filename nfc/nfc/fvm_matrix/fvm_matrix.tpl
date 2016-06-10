class ${name}:
  public nosh::fvm_matrix
{
  public:
    ${name}(
        ${constructor_args}
        ):
      ${members_init}
    {
      this->fill();
    }

    virtual
      ~${name}()
      {}

  private:
    ${members_declare}
}; // class ${name}
