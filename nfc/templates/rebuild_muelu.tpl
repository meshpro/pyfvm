if (MueluPrec_.is_null()) {
#ifdef NOSH_TEUCHOS_TIME_MONITOR
  Teuchos::TimeMonitor tm(*timerRebuild0_);
#endif
  Teuchos::ParameterList params;
  ${muelu_parameter_set}
  //params.set("number of equations", 2);
  //params.set("reuse: type", "full");

  // For some reason, we need to rcp explicitly. Otherwise, the call to
  // MueLu::CreateTpetraPreconditioner will complain about unmatching types.
  Teuchos::RCP<Tpetra::CrsMatrix<double,int,int>> rMatrixRcp =
    Teuchos::rcp(${matrix});

  MueluPrec_ = MueLu::CreateTpetraPreconditioner(
      rMatrixRcp,
      params
      );
} else {
#ifdef NOSH_TEUCHOS_TIME_MONITOR
  Teuchos::TimeMonitor tm(*timerRebuild1_);
#endif
  Teuchos::RCP<Tpetra::CrsMatrix<double,int,int>> rMatrixRcp =
    Teuchos::rcp(${matrix});
  MueLu::ReuseTpetraPreconditioner(
      rMatrixRcp,
      *MueluPrec_
      );
}
