# HW 4 Q 3
# Newman Exercise 5.13, P182
# Quantum uncertainty in the harmonic oscillator

import numpy as np
import math
import matplotlib.pyplot as plt
import quadrature
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
    return  pow(np.sin(x), 2) * np.exp(-pow(np.tan(x), 2))*pow(H(n, np.tan(x)), 2) \
    / (pow(2, n)* math.factorial(n) * math.sqrt(math.pi) * pow(np.cos(x), 4)) 
    

def qc():
    '''
    (c) uncertainty
    calculate gaussian quadrature
    '''
    N = 100
    xs, ws = quadrature.gaussxwab(N, -math.pi/2, math.pi/2)
    fs = integrandc(xs)
    res = math.sqrt(np.sum(ws * fs))
    print(f"(c) quantum uncertainty: {res}")

# qc()

def integrandd(x, n = 5):
    '''
    integrand of question d
    input n: energy level
    '''
    return  x * x / (pow(2, n)* math.factorial(n) * math.sqrt(math.pi)) * pow(H(n, x), 2)

def qd():
    '''
    uncertainty
    calculate Gauss-Hermite quadrature
    '''
    N = 100 # sample points
    xs, ws = roots_hermite(N)
    fs = integrandd(xs)
    res = math.sqrt(np.sum(ws * fs))
    print(f"(d) quantum uncertainty: {res}")

qd()