class ${name}: public Tpetra::Operator<double,int,int> {

public:

${name}(
    const std::shared_ptr<const nosh::mesh> & mesh,
    const Tpetra::Vector<double,int,int> & x0
    ) :
  ${members_init}
{
}

~${name}()
{
}

void
apply(
    const Tpetra::MultiVector<double,int,int> &x,
    Tpetra::MultiVector<double,int,int> &y,
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
${apply}
return;
}

Teuchos::RCP<const Tpetra::Map<int,int>>
getDomainMap() const
{
  return x0_->getMap();
}

Teuchos::RCP<const Tpetra::Map<int,int>>
getRangeMap() const
{
  return x0_->getMap();
}

void
rebuild(
    const Tpetra::Vector<double,int,int> & x0
    )
{
  x0_ = x0;
  return;
}

protected:
private:
  ${members}
} // class ${name}
