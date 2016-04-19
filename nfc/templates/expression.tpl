class ${name}:
  public nosh::expression
{
  public:
    ${name}():
      nosh::expression(0)
    {}

    virtual
    ~${name}()
    {}

    virtual
    double
    operator()(const Eigen::Vector3d & x) const
    {
      ${eval_body}
    };
};
