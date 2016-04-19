class ${name}Preconditioner: public Tpetra::Operator<double,int,int>
{
public:

${name}Preconditioner(
    const std::shared_ptr<const nosh::mesh> &mesh,
    const Tpetra::Vector<double,int,int> & x0
    ):
  mesh_(mesh)
  ,x0_(x0)
  ${init_code}
#ifdef NOSH_TEUCHOS_TIME_MONITOR
  ,timerRebuild0_(Teuchos::TimeMonitor::getNewTimer(
        "${name}Preconditioner::rebuild::MueLu init"
        ))
  ,timerRebuild1_(Teuchos::TimeMonitor::getNewTimer(
        "${name}Preconditioner::rebuild::MueLu rebuild"
        ))
#endif
{
}

~${name}Preconditioner() {}

void
apply(
    const Tpetra::MultiVector<double,int,int> &X,
    Tpetra::MultiVector<double,int,int> &Y,
    Teuchos::ETransp mode,
    double alpha,
    double beta
    ) const
{
  TEUCHOS_ASSERT_EQUALITY(mode, Teuchos::NO_TRANS);
  TEUCHOS_ASSERT_EQUALITY(alpha, 1.0);
  TEUCHOS_ASSERT_EQUALITY(beta, 0.0);

${apply_prec_code}

  else {
    // Belos part
    Teuchos::ParameterList belosList;
    // Relative convergence tolerance requested.
    // Set this to 0 and adapt the maximum number of iterations. This way, the
    // preconditioner always does exactly the same thing (namely maxIter PCG
    // iterations) and is independent of X. This avoids mathematical
    // difficulties.
    belosList.set("Convergence Tolerance", 0.0);
    belosList.set("Maximum Iterations", numCycles_);
//     belosList.set("Verbosity",
//                   Belos::Errors +
//                   Belos::Warnings +
//                   Belos::TimingDetails +
//                   Belos::StatusTestDetails
//                 );
//     belosList.set("Output Frequency", 10);
    belosList.set("Verbosity", Belos::Errors + Belos::Warnings);

    // Make sure to have a solid initial guess.
    // Belos, for example, does not initialize Y before passing it here.
    Y.putScalar(0.0);

    // Construct an unpreconditioned linear problem instance.
    auto Xptr = Teuchos::rcpFromRef(X);
    auto Yptr = Teuchos::rcpFromRef(Y);
    Belos::LinearProblem<double, MV, OP> problem(
        Teuchos::rcp(regularizedKeo_),
        Yptr,
        Xptr
        );
    // Make sure the problem sets up correctly.
    TEUCHOS_ASSERT(problem.setProblem());

    // add preconditioner
    // TODO recheck
    // Create the Belos preconditioned operator from the preconditioner.
    // NOTE:  This is necessary because Belos expects an operator to apply the
    //        preconditioner with Apply() NOT ApplyInverse().
    //Teuchos::RCP<Belos::EpetraPrecOp> mueluPrec =
    //  Teuchos::rcp(new Belos::EpetraPrecOp(MueluPrec_));
    problem.setLeftPrec(MueluPrec_);

    // Create an iterative solver manager.
    Belos::PseudoBlockCGSolMgr<double, MV, OP> newSolver(
        Teuchos::rcp(&problem, false),
        Teuchos::rcp(&belosList, false)
        );

    // Perform "solve".
    Belos::ReturnType ret = newSolver.solve();

    TEUCHOS_ASSERT_EQUALITY(ret, Belos::Converged);

    return;
  }
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
    const std::map<std::string, double> & params,
    const Tpetra::Vector<double,int,int> & x
    )
{
  // Copy over the matrix.
  // This is necessary as we don't apply AMG to K, but to K + g*2|psi|^2.
  // A possible work-around this copy would be to define the matrixBuilder's
  // matrix as K+2*g*|psi|^2 in the first place, and make sure that 2*g*|psi|^2
  // is taken away again wherever needed (e.g., the Jacobian).  This would
  // introduce the additional complication of having KEO depend on psi, and
  // would likely lead to some confusion in the rest of the code.  Hence, don't
  // worry too much about this until memory contrains get tight.
  regularizedKeo_->setParameters(params);

  const double g = params.at("g");
  // Add 2*g*|psi|^2 to the diagonal.
  if (g > 0.0) {
    // To be added to the diagonal blocks:
    //
    // [alpha + gamma, beta         ]
    // [beta,          alpha - gamma].
    //
    // We could also ahead and only add alpha to the diagonal, i.e.,
    //
    //const std::shared_ptr<const Tpetra::Vector<double,int,int>> absPsiSquared =
    //  this->getAbsPsiSquared_(x);
//#ifndef NDEBUG
    //TEUCHOS_ASSERT(regularizedKeo_.RowMap().SameAs(absPsiSquared->Map()));
//#endif
    //Tpetra::Vector<double,int,int> diag(regularizedKeo_.RowMap());
    //TEUCHOS_ASSERT_EQUALITY(0, regularizedKeo_.ExtractDiagonalCopy(diag));
    //TEUCHOS_ASSERT_EQUALITY(0, diag.Update(g*2.0, *absPsiSquared, 1.0));
    //TEUCHOS_ASSERT_EQUALITY(0, regularizedKeo_.ReplaceDiagonalValues(diag));
    //
    const auto & control_volumes = *(mesh_->getControlVolumes());
    const auto thicknessValues = thickness_->getV(params);
#ifndef NDEBUG
    TEUCHOS_ASSERT(control_volumes.getMap()->isSameAs(*thicknessValues.getMap()));
#endif
    auto cData = control_volumes.getData();
    auto tData = thicknessValues.getData();
    auto xData = x.getData();
#ifndef NDEBUG
    TEUCHOS_ASSERT_EQUALITY(cData.size(), tData.size());
    TEUCHOS_ASSERT_EQUALITY(2*tData.size(), xData.size());
#endif
    Teuchos::Tuple<int,2> idx;
    Teuchos::Tuple<double,2> vals;
    regularizedKeo_->resumeFill();
    for (int k = 0; k < cData.size(); k++) {
      const double alpha = g * cData[k] * tData[k]
        * 2.0 * (xData[2*k]*xData[2*k] + xData[2*k+1]*xData[2*k+1]);

      const double beta = g * cData[k] * tData[k]
                          * (2.0 * xData[2*k] * xData[2*k+1]);

      const double gamma = g * cData[k] * tData[k]
        * (xData[2*k]*xData[2*k] - xData[2*k+1]*xData[2*k+1]);

      // TODO check if the indices are correct here
      idx[0] = 2*k;
      idx[1] = 2*k + 1;
      vals[0] = alpha + gamma;
      vals[1] = beta;
      TEUCHOS_ASSERT_EQUALITY(
          2,
          regularizedKeo_->sumIntoLocalValues(2*k, idx, vals)
          );
      vals[0] = beta;
      vals[1] = alpha - gamma;
      TEUCHOS_ASSERT_EQUALITY(
          2,
          regularizedKeo_->sumIntoLocalValues(2*k+1, idx, vals)
          );
    }
  }
  regularizedKeo_->fillComplete();

${rebuild_code}

  return;
}

protected:
private:

${declare_code}
}
