# HW 5 Q 1
# Newman Exercise 5.17 The gamma function
# P 205

import numpy as np
import matplotlib.pyplot as plt
import quadrature

def integrand(x, a):
    return pow(x, a-1) * np.exp(-x)

def qa():
    alist = [2, 3, 4]
    xs = np.linspace(0, 5, 200)
    for a in alist:
        ys = integrand(xs, a)
        plt.plot(xs, ys, label = f"a = {a}")
    plt.title("Integrand")
    plt.xlabel("x")
    plt.ylabel("integrand")
    plt.legend()
    plt.savefig("integrand.png")

def integrand_new(z, a):
    exponent = (a-1)*np.log(z) - (a+1)*np.log(1-z) - (a-1)*z/(1-z)
    return np.exp(exponent)

def gamma(a):
    '''
    use gaussian quadrature to integrate
    '''
    N = 500 # roots of gaussian quadrature
    xs, ws = quadrature.gaussxwab(N, 0, 1)
    ys = integrand_new(xs, a)
    return np.sum(ws*ys)*pow(a-1, a)
    
def qe():
    res = gamma(1.5)
    print(f"gamma(1.5)={res}")

def qf():
    print(f"gamma(3)={gamma(3)}")
    print(f"gamma(6)={gamma(6)}")
    print(f"gamma(10)={gamma(10)}")

# qe()
qf()
