import pyfvm
import pytest
import sympy


a = sympy.Symbol('a')
x = sympy.Symbol('x')
y = sympy.Symbol('y')


@pytest.mark.parametrize('expr, parts', [
    (x, (0, [1, 0], 0)),
    (y, (0, [0, 1], 0)),
    (a, (a, [0, 0], 0)),
    (x**2, (0, [0, 0], x**2)),
    (x*y, (0, [0, 0], x*y)),
    ])
def test_split(expr, parts):
    pts = pyfvm.split(expr, [x, y])
    assert pts == parts
    return


if __name__ == '__main__':
    pts = pyfvm.split(a*(x - y), [x, y])
    print(pts)
