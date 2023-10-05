# HW 4 Q 2
# Newman Excercise 5.10, P 173
# Period of an anharmonic oscillator

import quadrature
import numpy as np
import matplotlib.pyplot as plt

m = 1 # mass

def potential(x):
    return pow(x, 4)

def integrand(x, a):
    return 1 / np.sqrt(potential(a) - potential(0))

def calc_period(a, N = 20):
    '''
    calculate period of oscillator
    a: amplitude 
    N: sample points of Gaussian Quadrature
    '''
    xs, ws = quadrature.gaussxwab(N, 0, a)
    fs = integrand(xs, a)
    return np.sum(ws * fs) * np.sqrt(8*m)

alist = np.linspace(0.01, 2, 100)
Ts = []
for a in alist:
    Ts.append(calc_period(a))
Ts = np.array(Ts)
plt.plot(alist, Ts)
plt.title("Period of Oscillator")
plt.xlabel("amplitude")
plt.ylabel("period")
plt.savefig("period.png")