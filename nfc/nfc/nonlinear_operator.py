# -*- coding: utf-8 -*-
#
class CodeNonlinearOperator(object):
    def __init__(self, F, name):
        self.F = F
        self.name = name

    def get_code(self):
        assert(isinstance(F, nfl.NonlinearProblem))

        debug = True
        if debug:
            logging.basicConfig(level=logging.DEBUG)

        # The dissection of nonlinear operators is unfortunately not so easy. We
        # cannot use SymPy's built-in C-code generation here, since the operator
        # may be composed of more complex operations than the basic ones. For
        # example, the operator could involve the application of linear operator
        # defined elsewhere.
        # For this reason, we're doing the code generation manually. This
        # involves
        #   (1) generating the abstract syntax tree, and
        #   (2) generating the code from the AST.
        #
        # Call the function and check the return value.
        generator = CodeGen()
        u = sympy.Symbol('u')
        f_code, required_ops = generator.generate(F.f(u), {'u': 'x'})
        print(f_code)

        print('-----------------------------')

        if F.dfdp:
            dfdp_code, required_ops = generator.generate(F.dfdp(u), {'u': 'x'})
            print(dfdp_code)
            print('-----------------------------')

        if F.jac:
            u0 = sympy.Symbol('u0')
            jac_code, required_ops = generator.generate(
                F.jac(u, u0),
                {'u': 'x', 'u0': 'x0_'}
                )
            print(jac_code)
            print('-----------------------------')

        print(F)
        if F.prec:
            prec_code = get_preconditioner_code(F.prec)
            print(prec_code)
            print('-----------------------------')

        print('lulz')
        exit()

        # TODO
        # Check if any of the arguments is not used in the function.
        # (We'll declare them (void) to supress compiler warnings.)
        # template substitution
        with open(os.path.join(templates_dir, 'ModelEvaluator.tpl'), 'r') as f:
            src = Template(f.read())
            code = src.substitute({
                'name': name.title(),
                'description': 'description',
                'compute_f_body': f_code,
                'compute_dfdp_body': dfdp_code,
                'belos_options_code': belos_options_code
                })

        print(code)

        with open(os.path.join(templates_dir, 'jacobian.tpl'), 'r') as f:
            src = Template(f.read())
            code = src.substitute({
                'name': name.title() + 'Jacobian',
                'description': 'description',
                'body': jac_code,
                })

        print(code)
        exit()
        return code


def to_c_string(val):
    if isinstance(val, str):
        return '"' + val + '"'
    else:
        return str(val)


def get_preconditioner_code(prec):
    op = prec(u0)
    assert(is_fvm_matrix(op))
    init_code = ',%s_(%s(mesh, x0))\n' \
        % (str(op), str(op).capitalize())
    print(init_matrix_code)

    if prec['solver']['type'] == 'Muelu':
        # Muelu-only preconditioner
        declare_prec_code, init_prec_code, rebuild_prec_code = \
            get_muelu_code(prec['solver']['parameters'])

        apply_prec_code = '''
#ifndef NDEBUG
  TEUCHOS_ASSERT(!MueluPrec_.is_null());
#endif
return MueluPrec_->apply(X, Y);
'''
    else:
        # Assume that we have a Belos Krylov method à la "Pseudo Block CG".
        parameters = ''
        for key, value in prec['solver']['parameters'].items():
            parameters += 'params.set("%s", %s)' % (key, to_c_string(value))

        set_preconditioner = ''
        if prec['solver']['preconditioner']:
            tp = prec['solver']['preconditioner']['type']
            if tp == 'Muelu':
                declare_code, init_code, rebuild_code = \
                    get_muelue_code(prec['solver']['parameters'])
            else:
                raise ValueError(
                    'Illegal Belos preconditioner type "%s".' % tp
                    )

        with open(os.path.join(templates_dir, 'Belos.tpl'), 'r') as f:
            src = Template(f.read())
            rebuild_code = src.substitute({
                'set_parameters': parameters,
                'operator': str(op),
                'set_preconditioner': set_preconditioner,
                'method': prec['solver']['type'].replace(' ', '')
                })

    with open(os.path.join(templates_dir, 'Preconditioner.tpl'), 'r') as f:
        src = Template(f.read())
        prec_code = src.substitute({
            'name': name.title(),
            'init_matrix': init_matrix_code,
            'apply_prec_code': apply_prec_code,
            'rebuild_code': rebuild_code
            })
    return prec_code


def get_muelu_code(parameters):
    declare_code = \
        'Teuchos::RCP<MueLu::TpetraOperator<double,int,int>> MueluPrec_;'
    init_code = 'MueluPrec_(Teuchos::null)'

    parameters = ''
    for key, value in prec['solver']['parameters'].items():
        parameters += 'params.set("%s", %s)' % (key, to_c_string(value))

    with open(os.path.join(templates_dir, 'RebuildMuelu.tpl'), 'r') as f:
        src = Template(f.read())
        rebuild_code = src.substitute({
            'muelu_parameter_set': parameters,
            'matrix': str(op) + '_'
            })

    return declare_code, init_code, rebuild_code

