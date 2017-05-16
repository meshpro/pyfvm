# -*- coding: utf-8 -*-
#
import numpy
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
    def __init__(self):
        self.is_boundary_only = True
        return

    def is_inside(self, x):
        return numpy.ones(x.shape[1], dtype=bool)


class Callable(object):
    def __init__(self, *args):
        self.args = args
        return


class FunctionParameter(sympy.Function):
    pass


class KernelList(object):
    '''A kernel is an entity that can occur in a the definition of `apply()` for
    an operator. That's either an Integral or a Kernel.
    The purpose of organizing them into a KernelList is to make it possible to
    "add" kernels, which eventually comes down to just collecting the kernels
    into a list.
    '''
    def __init__(self, integrals, kernels=None):
        self.integrals = integrals
        self.kernels = [] if kernels is None else kernels
        return

    def __add__(self, other):
        self.integrals.extend(other.integrals)
        self.kernels.extend(other.kernels)
        return self

    def __sub__(self, other):
        if other.kernels:
            raise NotImplementedError('Cannot negate kernels yet.')
        # flip the sign on the integrand of all 'other' kernels
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
        if self.kernels:
            raise NotImplementedError('Cannot negate kernels yet.')
        # flip the sign on the integrand of all 'self' kernels
        new_integrals = [Integral(
                lambda x: -integral.integrand(x),
                integral.measure,
                integral.subdomains
                ) for integral in self.integrals]
        self.integrals = new_integrals
        return self

    def __mul__(self, other):
        if self.kernels:
            raise NotImplementedError('Cannot multiply kernels yet.')
        assert(isinstance(other, float) or isinstance(other, int))
        # flip the sign on the integrand of all 'self' kernels
        new_integrals = [Integral(
                lambda x: other * integral.integrand(x),
                integral.measure,
                integral.subdomains
                ) for integral in self.integrals]
        self.integrals = new_integrals
        return self

    __rmul__ = __mul__


class FvmProblem(object):
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

    return KernelList([Integral(integrand, measure, subdomains)])


class Integral(KernelList):
    def __init__(self, integrand, measure, subdomains):
        self.integrand = integrand
        self.measure = measure
        self.subdomains = subdomains
        return


class EdgeKernel(object):
    pass


class dot(sympy.Function):
    pass


class n_dot_grad(sympy.Function):
    pass


n = sympy.MatrixSymbol('n', 3, 1)
neg_n = sympy.MatrixSymbol('neg_n', 3, 1)
