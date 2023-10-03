# HW 4 Q 3
# Newman Exercise 5.13, P182
# Quantum uncertainty in the harmonic oscillator

import numpy as np
import math
import matplotlib.pyplot as plt
# from heat_capacity import *
from scipy.special import roots_hermite

def H(n,x):
    '''
    Hermite Polynomial
    input x: array
    '''
    if n==0:
        return np.ones(x.shape)
    elif n==1:
        return 2*x
    else:
        return 2*x*H(n-1, x)-2*(n-1)*H(n-2, x)

def qa():  
    '''
    question a
    '''
    ns = [0, 1, 2, 3]
    xs = np.linspace(-4, 4, 100)
    for n in ns:
        hs = H(n, xs)
        plt.plot(xs, hs, label = f"n = {n}")
    plt.legend()
    plt.title("Hermite Polynomials")
    plt.xlabel("x")
    plt.ylabel("H(n, x)")
    plt.savefig("hermite.png")
    plt.clf()

# qa()

def qb():
    '''
    question b
    '''
    xs = np.linspace(-10, 10, 300)
    hs = H(30, xs)
    plt.plot(xs, hs)
    plt.title("Hermite Polynomial (n = 30)")
    plt.xlabel("x")
    plt.ylabel("H(30, x)")
    plt.savefig("hermite30.png")

# qb()

def integrandc(x, n = 5):
    '''
    integrand of question c
    input n: energy level
    '''
    return  x * x / (pow(2, n)* math.factorial(n) * math.sqrt(math.pi)) * pow(H(n, x), 2)

def qc():
    '''
    uncertainty
    calculate Gauss-Hermite quadrature
    '''
    N = 100 # sample points
    xs, ws = roots_hermite(N)
    fs = integrandc(xs)
    res = math.sqrt(np.sum(ws * fs))
    print(f"quantum uncertainty: {res}")

# qc()