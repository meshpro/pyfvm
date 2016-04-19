class ModelEvaluator:
  public nosh::ModelEvaluator
{
public:
ModelEvaluator(
    const std::shared_ptr<const nosh::mesh> & mesh,
    const std::shared_ptr<const Tpetra::Vector<double,int,int>> & x,
    const std::string & derivParameter
   ) :
  mesh_(mesh),
#ifdef NOSH_TEUCHOS_TIME_MONITOR
  evalModelTime_(Teuchos::TimeMonitor::getNewTimer(
        "${name}: evalModel"
        )),
  computeFTime_(Teuchos::TimeMonitor::getNewTimer(
        "${name}: evalModel:compute F"
        )),
  computedFdpTime_(Teuchos::TimeMonitor::getNewTimer(
        "${name}: evalModel:compute dF/dp"
        )),
  fillJacobianTime_(Teuchos::TimeMonitor::getNewTimer(
        "${name}: evalModel:fill Jacobian"
        )),
  fillPreconditionerTime_(Teuchos::TimeMonitor::getNewTimer(
        "${name}: evalModel::fill preconditioner"
        )),
#endif
  out_(Teuchos::VerboseObjectBase::getDefaultOStream()),
  p_map_(Teuchos::null),
  p_names_(Teuchos::null),
  nominalValues_(this->createInArgs()),
  space_(x->space())
{
  // Merge all of the parameters together.
  std::map<std::string, double> params;

  // This merges and discards new values if their keys are already in the list.
  auto spParams = scalarPotential_->getParameters();
  params.insert(spParams.begin(), spParams.end());

  // This merges and discards new values if their keys are already in the list.
  auto mbParams = keo_->getParameters();
  params.insert(mbParams.begin(), mbParams.end());

  // Out of this now complete list, create the entities that the Modelevaluator
  // needs.
  const int numParams = params.size();
  p_map_ = Teuchos::rcp(
      new Tpetra::Map<int,int>(
        numParams,
        0,
        Teuchos::rcp(mesh_->comm)
        )
      );
  auto p_init = Thyra::createMember(this->get_p_space(0));
  p_names_ = Teuchos::rcp(new Teuchos::Array<std::string>(numParams));
  int k = 0;
  for (auto it = params.begin(); it != params.end(); ++it) {
    (*p_names_)[k] = it->first;
    Thyra::set_ele(k, it->second, p_init());
    k++;
  }

  // set nominal values
  auto xxx = Thyra::createConstVector(Teuchos::rcp(initialX), space_);
  nominalValues_.set_p(0, p_init);
  nominalValues_.set_x(xxx);

  return;
}
// ============================================================================
~ModelEvaluator()
{
}
// ============================================================================
Teuchos::RCP<const Thyra::VectorSpaceBase<double>>
get_x_space() const
{
#ifndef NDEBUG
  TEUCHOS_ASSERT(!space_.is_null());
#endif
  return space_;
}
// ============================================================================
Teuchos::RCP<const Thyra::VectorSpaceBase<double>>
get_f_space() const
{
#ifndef NDEBUG
  TEUCHOS_ASSERT(!space_.is_null());
#endif
  return space_;
}
// ============================================================================
Teuchos::RCP<const Thyra::VectorSpaceBase<double>>
get_p_space(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPT_MSG(
      l != 0,
      "LOCA can only deal with one parameter vector."
      );
  return Thyra::createVectorSpace<double>(p_map_);
}
// ============================================================================
Teuchos::RCP<const Teuchos::Array<std::string>>
get_p_names(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPT_MSG(
      l != 0,
      "LOCA can only deal with one parameter vector."
      );
  return p_names_;
}
// =============================================================================
Teuchos::RCP<const Thyra::VectorSpaceBase<double>>
get_g_space(int l) const
{
  (void) l;
  TEUCHOS_TEST_FOR_EXCEPT_MSG(
      true,
      "Not implemented."
      );
  return Teuchos::null;
}
// =============================================================================
Thyra::ModelEvaluatorBase::InArgs<double>
getNominalValues() const
{
  return nominalValues_;
}
// =============================================================================
Thyra::ModelEvaluatorBase::InArgs<double>
getLowerBounds() const
{
  TEUCHOS_TEST_FOR_EXCEPT_MSG(
      true,
      "Not implemented."
      );
  return this->createInArgs();
}
// =============================================================================
Thyra::ModelEvaluatorBase::InArgs<double>
getUpperBounds() const
{
  TEUCHOS_TEST_FOR_EXCEPT_MSG(
      true,
      "Not implemented."
      );
  return this->createInArgs();
}
// =============================================================================
Teuchos::RCP<Thyra::LinearOpBase<double>>
create_W_op() const
{
  Tpetra::Vector<double,int,int> x0(this->space_);
  Teuchos::RCP<Tpetra::Operator<double,int,int>> jac = Teuchos::rcp(
        new ${name}::JacobianOperator(
          mesh_,
          x0
          )
        );

  return Thyra::createLinearOp(jac, space_, space_);
}
// =============================================================================
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<double>>
get_W_factory() const
{
  Stratimikos::DefaultLinearSolverBuilder builder;

  auto p = Teuchos::rcp(new Teuchos::ParameterList);
  p->set("Linear Solver Type", "Belos");
  auto & belosList =
    p->sublist("Linear Solver Types")
    .sublist("Belos");

${belos_options_code}

  p->set("Preconditioner Type", "None");
  builder.setParameterList(p);

  auto lowsFactory = builder.createLinearSolveStrategy("");

  lowsFactory->setVerbLevel(Teuchos::VERB_LOW);

  return lowsFactory;
}
// =============================================================================
Teuchos::RCP<Thyra::PreconditionerBase<double>>
create_W_prec() const
{
  const Teuchos::RCP<Tpetra::Operator<double,int,int>> keoPrec = Teuchos::rcp(
      new ${name}::JacPreconditioner(
        mesh_
        )
      );
  auto keoT = Thyra::createLinearOp(keoPrec, space_, space_);
  return Thyra::nonconstUnspecifiedPrec(keoT);
}
// ============================================================================
void
reportFinalPoint(
    const Thyra::ModelEvaluatorBase::InArgs<double> &finalPoint,
    const bool wasSolved
    )
{
  (void) finalPoint;
  (void) wasSolved;
  TEUCHOS_TEST_FOR_EXCEPT_MSG(
      true,
      "Not implemented."
      );
}
// ============================================================================
Thyra::ModelEvaluatorBase::InArgs<double>
createInArgs() const
{
  Thyra::ModelEvaluatorBase::InArgsSetup<double> inArgs;

  inArgs.setModelEvalDescription(${description});

  // We have *one* parameter vector with numParams_ parameters in it.
  inArgs.set_Np(1);

  inArgs.setSupports(IN_ARG_x, true);

  // for shifted matrix
  // TODO add support for operator shift
  inArgs.setSupports(IN_ARG_alpha, true);
  inArgs.setSupports(IN_ARG_beta, true);

  return inArgs;
}
// ============================================================================
Thyra::ModelEvaluatorBase::OutArgs<double>
createOutArgsImpl() const
{
  Thyra::ModelEvaluatorBase::OutArgsSetup<double> outArgs;

  outArgs.setModelEvalDescription(${description});

  outArgs.set_Np_Ng(1, 0); // one parameter vector, no objective function

  outArgs.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_f);

  // support derivatives with respect to all parameters;
  outArgs.setSupports(
      Thyra::ModelEvaluatorBase::OUT_ARG_DfDp,
      0,
      DerivativeSupport(DERIV_MV_BY_COL)
      );

  outArgs.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_op);
  //outArgs.set_W_properties(
  //    DerivativeProperties(
  //      DERIV_LINEARITY_UNKNOWN, // DERIV_LINEARITY_NONCONST
  //      DERIV_RANK_DEFICIENT, // DERIV_RANK_FULL, DERIV_RANK_DEFICIENT
  //      false // supportsAdjoint
  //      )
  //    );

  outArgs.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_prec);
  //outArgs.set_W_prec_properties(
  //    DerivativeProperties(
  //      DERIV_LINEARITY_UNKNOWN,
  //      DERIV_RANK_FULL,
  //      false
  //      )
  //    );

  return outArgs;
}
// ============================================================================
void
evalModelImpl(
    const Thyra::ModelEvaluatorBase::InArgs<double> & inArgs,
    const Thyra::ModelEvaluatorBase::OutArgs<double> & outArgs
    ) const
{
#ifdef NOSH_TEUCHOS_TIME_MONITOR
  Teuchos::TimeMonitor tm0(*evalModelTime_);
#endif

  const double alpha = inArgs.get_alpha();
  double beta = inArgs.get_beta();

  // From packages/piro/test/MockModelEval_A.cpp
  if (alpha == 0.0 && beta == 0.0) {
    beta = 1.0;
  }
#ifndef NDEBUG
  TEUCHOS_ASSERT_EQUALITY(alpha, 0.0);
  TEUCHOS_ASSERT_EQUALITY(beta,  1.0);
#endif

  const auto & x_in = inArgs.get_x();
#ifndef NDEBUG
  TEUCHOS_ASSERT(!x_in.is_null());
#endif
  // create corresponding tpetra vector
  auto x_in_tpetra =
    Thyra::TpetraOperatorVectorExtraction<double,int,int>::getConstTpetraVector(
        x_in
        );

  // Dissect inArgs.get_p(0) into parameter sublists.
  const auto & p_in = inArgs.get_p(0);
#ifndef NDEBUG
  TEUCHOS_ASSERT(!p_in.is_null());
#endif

#ifndef NDEBUG
  // Make sure the parameters aren't NaNs.
  for (int k = 0; k < p_in->space()->dim(); k++) {
    TEUCHOS_ASSERT(!std::isnan(Thyra::get_ele(*p_in, k)));
  }
#endif

  // Fill the parameters into a std::map.
  const auto paramNames = this->get_p_names(0);
  std::map<std::string, double> params;
  for (int k = 0; k < p_in->space()->dim(); k++) {
    params[(*paramNames)[k]] = Thyra::get_ele(*p_in, k);
    //std::cout << (*paramNames)[k] << " " << (*p_in)[k] << std::endl;
  }

  // compute F
  const auto & f_out = outArgs.get_f();
  if (!f_out.is_null()) {
#ifdef NOSH_TEUCHOS_TIME_MONITOR
    Teuchos::TimeMonitor tm1(*computeFTime_);
#endif

    auto f_out_tpetra =
      Thyra::TpetraOperatorVectorExtraction<double,int,int>::getTpetraVector(
          f_out
          );
    this->computeF_(
        *x_in_tpetra,
        params,
        *f_out_tpetra
        );
  }

  // Compute df/dp.
  const auto & derivMv = outArgs.get_DfDp(0).getDerivativeMultiVector();
  const auto & dfdp_out = derivMv.getMultiVector();
  if (!dfdp_out.is_null()) {
#ifdef NOSH_TEUCHOS_TIME_MONITOR
    Teuchos::TimeMonitor tm2(*computedFdpTime_);
#endif
    auto dfdp_out_tpetra =
      Thyra::TpetraOperatorVectorExtraction<double,int,int>::getTpetraMultiVector(
          dfdp_out
          );

    const int numAllParams = this->get_p_space(0)->dim();
    TEUCHOS_ASSERT_EQUALITY(
        numAllParams,
        dfdp_out_tpetra->getNumVectors()
        );
    // Compute all derivatives.
    for (int k = 0; k < numAllParams; k++) {
      this->computeDFDP_(
          *x_in_tpetra,
          params,
          (*paramNames)[k],
          *dfdp_out_tpetra->getVectorNonConst(k)
          );
    }
  }

  // Fill Jacobian.
  const auto & W_out = outArgs.get_W_op();
  if(!W_out.is_null()) {
#ifdef NOSH_TEUCHOS_TIME_MONITOR
    Teuchos::TimeMonitor tm3(*fillJacobianTime_);
#endif
    auto W_outT =
      Thyra::TpetraOperatorVectorExtraction<double,int,int>::getTpetraOperator(
          W_out
          );
    const auto & jac =
      Teuchos::rcp_dynamic_cast<${name}::JacobianOperator>(W_outT, true);
    jac->rebuild(params, *x_in_tpetra);
  }

  // Fill preconditioner.
  const auto & WPrec_out = outArgs.get_W_prec();
  if(!WPrec_out.is_null()) {
#ifdef NOSH_TEUCHOS_TIME_MONITOR
    Teuchos::TimeMonitor tm4(*fillPreconditionerTime_);
#endif
    auto WPrec_outT =
      Thyra::TpetraOperatorVectorExtraction<double,int,int>::getTpetraOperator(
          WPrec_out->getNonconstUnspecifiedPrecOp()
          );
    const auto & prec =
      Teuchos::rcp_dynamic_cast<${name}::JacPreconditioner>(WPrec_outT, true);
    prec->rebuild(
        params,
        *x_in_tpetra
        );
  }

  return;
}
// ============================================================================
void
computeF_(
    const Tpetra::Vector<double,int,int> & x,
    const std::map<std::string, double> & params,
    Tpetra::Vector<double,int,int> & y
    ) const
{
${compute_f_body}
return;
}
// ============================================================================
void
computeDFDP_(
    const Tpetra::Vector<double,int,int> & x,
    const std::map<std::string, double> & params,
    const std::string & paramName,
    Tpetra::Vector<double,int,int> & y
    ) const
{
  ${compute_dfdp_body}
  return;
}
// =============================================================================
protected:
private:
} // class ModelEvaluator
