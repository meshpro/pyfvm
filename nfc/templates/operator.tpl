class ${light_class_name}: public Tpetra::Operator<double,int,int> {
public:
${light_class_name}(
    ${light_constructor_args}
    )${light_members_init}
{
}

virtual
~${light_class_name}()
{
}

void
apply(
    const Tpetra::MultiVector<double,int,int> & x,
    Tpetra::MultiVector<double,int,int> & y,
    Teuchos::ETransp mode,
    double alpha,
    double beta
    ) const
{
TEUCHOS_TEST_FOR_EXCEPT_MSG(
    mode != Teuchos::NO_TRANS,
    "Only untransposed applies supported."
    );
TEUCHOS_TEST_FOR_EXCEPT_MSG(
    alpha != 1.0,
    "Only alpha==1.0 supported."
    );
TEUCHOS_TEST_FOR_EXCEPT_MSG(
    beta != 0.0,
    "Only beta==0.0 supported."
    )
${light_apply}
${boundary_code}
return;
}

Teuchos::RCP<const Tpetra::Map<int,int>>
getDomainMap() const
{
  return mesh_->map;
}

Teuchos::RCP<const Tpetra::Map<int,int>>
getRangeMap() const
{
  return mesh_->map;
}

protected:
private:
${light_members_declare}
} // class ${light_class_name}


class ${full_class_name}: public Tpetra::Operator<double,int,int> {
public:
${full_class_name}(
    const std::shared_ptr<const nosh::mesh> & mesh
    ):
  ${full_members_init}
{
}

virtual
~${full_class_name}()
{
}

void
apply(
    const Tpetra::MultiVector<double,int,int> & x,
    Tpetra::MultiVector<double,int,int> & y,
    Teuchos::ETransp mode,
    double alpha,
    double beta
    ) const
{
  this->${light_var_name}->apply(x, y, mode, alpha, beta);
  return;
}

Teuchos::RCP<const Tpetra::Map<int,int>>
getDomainMap() const
{
  return this->${light_var_name}->getDomainMap();
}

Teuchos::RCP<const Tpetra::Map<int,int>>
getRangeMap() const
{
  return this->${light_var_name}->getRangeMap();
}

protected:
private:
  ${full_members_declare}
} // class ${full_class_name}
