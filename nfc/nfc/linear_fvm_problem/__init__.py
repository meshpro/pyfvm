# -*- coding: utf-8 -*-
#
from ..dirichlet import *
from ..integral_boundary import *
from ..integral_edge import *
from ..integral_vertex import *
from ..helpers import sanitize_identifier
from ..fvm_matrix import gather_core_dependencies, get_code_linear_problem


class LinearFvmProblemCode(object):
    def __init__(self, namespace, cls):
        self.class_name = sanitize_identifier(cls.__name__)
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
        code = get_code_linear_problem(
            'linear_fvm_problem.tpl',
            self.class_name,
            'nosh::linear_problem',
            self.dependencies
            )

        return {
            'code': code
            }

    def get_python_class_object(self, dep_class_objects):
        filename = os.path.join(os.path.dirname(__file__), 'python.tpl')
        code = get_code_linear_problem(
            filename,
            self.class_name,
            'nosh::linear_problem',
            self.dependencies
            )

        return {
            'code': code
            }
