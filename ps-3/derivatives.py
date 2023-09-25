# HW 3 Q 1
# Newman P 133 Excercise 4.3 Calculating derivatives

def func(_x):
    return _x * (_x - 1)

def func_derivative(_x):
    '''
    actual derivative
    '''
    return 2 * _x - 1

def calc_derivative(_x, _delta):
    '''
    derivative by the limits formula
    '''
    return (func(_x + _delta) - func(_x))/_delta

x = 1
a_derivative_limits = calc_derivative(x, 0.01)
a_derivative_analytically = func_derivative(1)
print("question (a):")
print(f"derivative by limits: {a_derivative_limits}, delta = 0.01")
print(f"derivative analytically: {a_derivative_analytically}")
print("question (b):")
delta_list = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
for delta in delta_list:
    div = calc_derivative(x, delta)
    print(f"derivative by limits: {div}, delta = {delta}")
