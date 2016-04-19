    Teuchos::ParameterList parameters;

    ${set_parameters}

    // Make sure to have a solid initial guess.
    // Belos, for example, does not initialize Y before passing it here.
    Y.putScalar(0.0);

    // Construct an unpreconditioned linear problem instance.
    auto Xptr = Teuchos::rcpFromRef(X);
    auto Yptr = Teuchos::rcpFromRef(Y);
    Belos::LinearProblem<double, MV, OP> problem(
        Teuchos::rcp(${operator}),
        Yptr,
        Xptr
        );
    // Make sure the problem sets up correctly.
    TEUCHOS_ASSERT(problem.setProblem());

    ${set_preconditioner}

    // Create an iterative solver manager.
    Belos::${method}SolMgr<double, MV, OP> newSolver(
        Teuchos::rcp(&problem, false),
        Teuchos::rcp(&parameters, false)
        );

    // Perform "solve".
    Belos::ReturnType ret = newSolver.solve();

    TEUCHOS_ASSERT_EQUALITY(ret, Belos::Converged);

    return;
  }
