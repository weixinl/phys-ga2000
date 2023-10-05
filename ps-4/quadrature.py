# HW 4 Q 1
# Newman Excercise 5.9, P 172
# Heat Capacity of a solid

import numpy as np
import matplotlib.pyplot as plt

######################################################################
# http://www-personal.umich.edu/~mejn/computational-physics/gaussxw.py
# Functions to calculate integration points and weights for Gaussian
# quadrature
#
# x,w = gaussxw(N) returns integration points x and integration
#           weights w such that sum_i w[i]*f(x[i]) is the Nth-order
#           Gaussian approximation to the integral int_{-1}^1 f(x) dx
# x,w = gaussxwab(N,a,b) returns integration points and weights
#           mapped to the interval [a,b], so that sum_i w[i]*f(x[i])
#           is the Nth-order Gaussian approximation to the integral
#           int_a^b f(x) dx
#
# This code finds the zeros of the nth Legendre polynomial using
# Newton's method, starting from the approximation given in Abramowitz
# and Stegun 22.16.6.  The Legendre polynomial itself is evaluated
# using the recurrence relation given in Abramowitz and Stegun
# 22.7.10.  The function has been checked against other sources for
# values of N up to 1000.  It is compatible with version 2 and version
# 3 of Python.
#
# Written by Mark Newman <mejn@umich.edu>, June 4, 2011
# You may use, share, or modify this file freely
#
######################################################################

from numpy import tan,pi

def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

############################################################################################################
#begin my code

v = 1e-3 # volume, m^3
rho = 6.022 * 1e28 # density, m^-3
theta = 428 # Debye temperature, k
kb = 1.380649 * 1e-23 # Boltzmann constant, J*K^-1

def integrand(x):
    return pow(x, 4) * np.exp(x)/pow(np.exp(x) - 1, 2)

def cv(T, N=50):
    '''
    heat capacity
    '''
    xs, ws = gaussxwab(N, 0, theta/T)
    fs = integrand(xs)
    return np.sum(ws * fs) * 9 * v * rho * kb * pow(T/theta, 3)

def qb():
    Ts = np.linspace(5, 500, 496) # T from 5K to 500K, 496 steps
    cvs = []
    for T in Ts:
        cvs.append(cv(T))
    cvs = np.array(cvs)
    plt.plot(Ts, cvs)
    plt.xlabel("T(K)")
    plt.ylabel("heat capacity (J/K)")
    plt.title("Heat Capacity to Temperature, N = 50")
    plt.savefig("heatcapacity50.png")
    plt.clf()

def qc():
    Ts = np.linspace(5, 500, 496) # T from 5K to 500K, 496 steps
    Ns = [10, 20, 30, 40, 50, 60, 70]
    for N in Ns:
        cvs = []
        for T in Ts:
            cvs.append(cv(T, N))
        cvs = np.array(cvs)
        plt.plot(Ts, cvs, label = f"N = {N}")
    plt.xlabel("T(K)")
    plt.ylabel("heat capacity (J/K)")
    plt.legend()
    plt.title("Heat Capacity to Temperature")
    plt.savefig("heatcapacity.png")   
    plt.clf()

if __name__ == "__main__":
    # qb()
    qc()
