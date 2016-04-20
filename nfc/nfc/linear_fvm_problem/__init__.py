# -*- coding: utf-8 -*-
#
from ..dirichlet import *
from ..integral_boundary import *
from ..integral_edge import *
from ..integral_vertex import *
from ..helpers import sanitize_identifier_cxx
from ..fvm_matrix import \
        gather_core_dependencies, \
        get_code_linear_problem_cxx, \
        get_code_linear_problem_python


class LinearFvmProblemCode(object):
    def __init__(self, namespace, cls):
        self.class_name_cxx = sanitize_identifier_cxx(cls.__name__)
        self.class_name_python = cls.__name__
        self.namespace = namespace

        u = sympy.Function('u')
        u.nosh = True

        res = cls.apply(u)
        self.dependencies = \
            gather_core_dependencies(
                    namespace, res, cls.dirichlet, matrix_var=u
                    )
        return

    def get_dependencies(self):
        return self.dependencies

    def get_cxx_class_object(self, dep_class_objects):
        filename = os.path.join(
                os.path.dirname(__file__),
                'linear_fvm_problem.tpl'
                )
        code = get_code_linear_problem_cxx(
            'linear_fvm_problem.tpl',
            self.class_name_cxx,
            'nosh::linear_problem',
            self.dependencies
            )

        return {
            'code': code
            }

    def get_python_class_object(self, dep_class_objects):
        filename = os.path.join(os.path.dirname(__file__), 'python.tpl')
        code = get_code_linear_problem_python(
            filename,
            self.class_name_python,
            self.dependencies
            )

        return {
            'code': code
            }
