class ${name}: public nosh::subdomain
{
  public:
    ${name}(): nosh::subdomain(${id}, ${is_boundary_only}) {}

    virtual ~${name}() {}

    virtual bool
    is_inside(const Eigen::Vector3d & x) const {
      ${is_inside_body}return ${is_inside_return};
    }
};  // class ${name}
