# Newman excercise 9.8 p 439
# The Schrodinger equation and the Crank-Nicolson method

import numpy as np
import banded
from copy import copy
import matplotlib.pyplot as plt

planck = 1.054571817e-34 # reduced planck constant

def plot_states(states, n, a, t):
    '''
    n: dimension of states
    a: side of a grid
    t: current time
    '''
    xlist = np.array(range(1, n + 1)) * a
    plt.plot(xlist, np.real(states))
    plt.xlabel("x [m]")
    plt.ylabel("$|\psi|$")
    plt.title(f"states at time {t}")
    plt.show()

def cranknicolson(init_states, n, a, h, m, steps = 1):
    '''
    init_states: initial state of each grid point at time 0
    n: dimension of init_states
    a: side of a grid
    h: time step
    m: mass
    steps: number of time steps
    '''
    a1 = 1 + h * 1j * planck / (2 * m * a * a)
    a2 = -h * 1j * planck / (4 * m * a * a)
    A_3row = np.zeros((3, n), dtype = np.complex_)
    for i in range(n):
        A_3row[1, i] = a1
    for i in range(1, n):
        A_3row[0, i] = a2
    for i in range(n - 1):
        A_3row[2, i] = a2
    b1 = 1 - h * 1j * planck / (2 * m * a * a)
    b2 = h * 1j * planck / (4 * m * a * a)
    B = np.zeros((n, n), dtype = np.complex_)
    for i in range(n):
        B[i, i] = b1
    for i in range(n - 1):
        B[i, i + 1] = b2
        B[i + 1, i] = b2
    states = init_states
    for step_i in range(steps):
        states = banded.banded(A_3row, np.matmul(B, copy(states)), 1, 1)
    return states


if __name__ == "__main__":
    L = 1e-8 # length of the box
    N = 1000 # spatial slices
    a = L/N # side of a grid
    x0 = L/2 # parameter of the initial wavefunction
    sigma = 1e-10 # parameter of the initial wavefunction
    k = 5e10 # parameter of the initial wavefunction
    m = 9.109e-31 # mass of an electron
    h = 1e-18 # time step
    init_states = np.zeros( N-1, dtype=np.complex_) # initial states at time 0 (between the wall)
    for i in range(N - 1):
        x = (i+1) * a
        init_states[i] = np.exp(-pow(x - x0, 2)/(2 * sigma * sigma)) *  np.exp(1j * k * x)
    steps = 500
    states = cranknicolson(init_states, N - 1, a, h, m, steps)
    plot_states(init_states, N - 1, a, 0)
    plot_states(states, N - 1, a, h * steps)