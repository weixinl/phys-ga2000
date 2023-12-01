# ps 8 q 2
# Newman Excercise 8.3  P 347
# The Lorenz equations

import numpy as np
import scipy
import matplotlib.pyplot as plt

def derivative_lorenz(t,r):
    """
    r = [x, y, z] 
    return their derivatives using lorenz equations
    """
    sigma_ = 10
    r_=28
    b_=8/3
    x = r[0]
    y = r[1]
    z = r[2]
    fx = sigma_*(y-x)
    fy = r_*x - y - x*z
    fz = x*y - b_*z
    return [fx, fy, fz] 

def numerical_traj_ex(t_span, y0, t):
    """
    t_span: time range
    y0: initial conditions
    t: specified time list
    """
    #deltaomega = 2*np.pi*deltaf
    sol4 = scipy.integrate.solve_ivp(derivative_lorenz, t_span, y0, t_eval = t,  \
                     method = 'LSODA')
    # Radau and LSODA yield the same results

    t = sol4.t
    y = sol4.y
    return t, y[0,:], y[1,:], y[2,:]

exp_fps = 1000 # samples per second
t_span = [0, 50] # simulate system for 50 seconds
t = np.arange(*t_span, 1/exp_fps)
y0 = [0,1,0] # initial condition
t, x, y, z = numerical_traj_ex(t_span, y0, t)

plt.plot(t, y)
plt.title("y-t")
plt.xlabel("t")
plt.ylabel("y")
plt.savefig("imgs/lorenz-yt.png")
plt.clf()

plt.plot(x, z)
plt.title("z-x")
plt.xlabel("x")
plt.ylabel("z")
plt.savefig("imgs/lorenz-zx.png")
plt.clf()