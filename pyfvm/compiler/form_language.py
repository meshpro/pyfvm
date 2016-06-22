# -*- coding: utf-8 -*-
#
import sympy


class FvmOperator(sympy.Function):
    pass


class Expression(sympy.Function):
    pass
    # degree = sympy.oo


class ScalarParameter(sympy.Symbol):
    pass


class Subdomain(object):
    pass


class Boundary(Subdomain):
    pass


class Callable(object):
    def __init__(self, *args):
        self.args = args
        return


class FunctionParameter(sympy.Function):
    pass


class CoreList(object):
    '''A core is an entity that can occur in a the definition of `apply()` for
    an operator. That's either an Integral or an FvmMatrix.
    The purpose of organizing them into a CoreList is to make it possible to
    "add" cores, which eventually comes down to just collecting the cores into
    a list.
    '''
    def __init__(self, integrals, fvm_matrices=None):
        self.integrals = integrals
        if fvm_matrices is None:
            fvm_matrices = []
        self.fvm_matrices = fvm_matrices

    def __add__(self, other):
        self.integrals.extend(other.integrals)
        self.fvm_matrices.extend(other.fvm_matrices)
        return self

    def __sub__(self, other):
        if other.fvm_matrices:
            raise NotImplementedError('Cannot negate FvmMatrices yet.')
        # flip the sign on the integrand of all 'other' cores
        new_integrals = [Integral(
                lambda x: -integral.integrand(x),
                integral.measure,
                integral.subdomains
                ) for integral in other.integrals]
        self.integrals.extend(new_integrals)
        return self

    def __pos__(self):
        return self

    def __neg__(self):
        if self.fvm_matrices:
            raise NotImplementedError('Cannot negate FvmMatrices yet.')
        # flip the sign on the integrand of all 'self' cores
        new_integrals = [Integral(
                lambda x: -integral.integrand(x),
                integral.measure,
                integral.subdomains
                ) for integral in self.integrals]
        self.integrals = new_integrals
        return self

    def __mul__(self, other):
        if self.fvm_matrices:
            raise NotImplementedError('Cannot multiply FvmMatrices yet.')
        assert(isinstance(other, float) or isinstance(other, int))
        # flip the sign on the integrand of all 'self' cores
        new_integrals = [Integral(
                lambda x: other * integral.integrand(x),
                integral.measure,
                integral.subdomains
                ) for integral in self.integrals]
        self.integrals = new_integrals
        return self

    __rmul__ = __mul__


class FvmMatrix(Callable, CoreList):
    # By default: No Dirichlet conditions.
    dirichlet = []

    def __init__(self, arg):
        CoreList.__init__(self, [], [self])
        return


class LinearFvmProblem(object):
    pass


class Measure(object):
    pass


class ControlVolume(Measure):
    pass

dV = ControlVolume()


class ControlVolumeSurface(Measure):
    pass

dS = ControlVolumeSurface()


class BoundarySurface(Measure):
    pass

dGamma = BoundarySurface()


def integrate(integrand, measure, subdomains=None):
    assert(isinstance(measure, Measure))

    if subdomains is None:
        subdomains = set()
    elif not isinstance(subdomains, set):
        try:
            subdomains = set(subdomains)
        except TypeError:  # TypeError: 'D1' object is not iterable
            subdomains = set([subdomains])

    assert(
        isinstance(measure, ControlVolumeSurface) or
        isinstance(measure, ControlVolume) or
        isinstance(measure, BoundarySurface)
        )

    return CoreList([Integral(integrand, measure, subdomains)])


class Integral(CoreList):
    def __init__(self, integrand, measure, subdomains):
        self.integrand = integrand
        self.measure = measure
        self.subdomains = subdomains
        return


class EdgeCore(object):
    pass


class dot(sympy.Function):
    pass


class n_dot_grad(sympy.Function):
    pass

n = sympy.MatrixSymbol('n', 3, 1)
neg_n = sympy.MatrixSymbol('neg_n', 3, 1)


def grad(a):
    return Vector()
