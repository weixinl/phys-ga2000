# HW2 Q4 Quadratic Equations
# Newman Excercise 4.2 P133
import numpy as np

def quadratic_a(_a, _b, _c):
    '''
    question a
    '''
    x_sqrt = np.sqrt(_b * _b - 4 * _a * _c)
    return (-_b + x_sqrt)/(2 * _a), (-_b - x_sqrt)/(2 * _a)

def quadratic_b(_a, _b, _c):
    '''
    question b
    '''
    x_sqrt = np.sqrt(_b * _b - 4 * _a * _c)
    return 2 * _c / (-_b - x_sqrt), 2 * _c / (-_b + x_sqrt)

def quadratic(a, b, c):
    '''
    solution with high precision
    '''
    x_sqrt = np.sqrt(b * b - 4 * a * c)
    if(abs(-b - x_sqrt) > abs(-b + x_sqrt)):
        return 2 * c / (-b - x_sqrt), (-b - x_sqrt)/(2 * a)
    else:
        return (-b + x_sqrt)/(2 * a), 2 * c / (-b + x_sqrt)


def func(_a, _b, _c, _x):
    return _a*_x*_x + _b*_x + _c

if __name__ == "__main__":
    a = 0.001
    b = 1000
    c = 0.001
    x1a, x2a = quadratic_a(a, b, c)
    print(f"(a) x1: {x1a}, x2: {x2a}")
    # print(func(a,b,c, x1a))
    # print(func(a,b,c, x2a))
    x1b, x2b = quadratic_b(a, b, c)
    print(f"(b) x1: {x1b}, x2: {x2b}")
    # print(func(a,b,c, x1b))
    # print(func(a,b,c, x2b))
    x1c, x2c = quadratic(a, b, c)
    print(f"(c) x1: {x1c}, x2: {x2c}")