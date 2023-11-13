import numpy as np
import matplotlib.pyplot as plt
import scipy
import timeit

golden_ratio = 0.38197
tol = 1e-7 # tolerance
epsilon = 1e-7 #for numerical stability

def f(x):
    '''
    test function
    '''
    return pow(x - 0.3, 2) * np.exp(x)

def parabolic(a, b, c):
    '''
    parabolic interpolation
    return min point of parabolic fit
    c is between a and b
    '''
    numerator = pow(b - a, 2) * (f(b) - f(c)) - pow(b - c, 2) * (f(b) - f(a))
    denominator = (b - a) * (f(b) - f(c)) - (b - c) * (f(b) - f(a)) + epsilon
    return b - 0.5 * numerator / denominator

def golden_section(a, b):
    '''
    golden section search for minimum
    initial bracket is (a, b)
    return minimum abscissa
    '''
    while(b - a > tol):
        c = a + (b - a) * golden_ratio # (a, c) interval is smaller than (c, b) interval
        x = c + (b - c) * golden_ratio
        fc = f(c)
        fx = f(x)
        if fc < fx:
            # a, c, x
            b = x
        else:
            # c, x, b
            a = c
            c = x
    return c


def brent():
    #define interval
    a = -10 
    b = 10
    c = (a + b)/2
    last_step = b - a
    second_last_step = last_step
    current_step = last_step
    tol = 1e-7
    while(abs(a - b) > tol):
        second_last_step = last_step
        last_step = current_step
        d = parabolic(a, b, c)
        if(d <= a or d >= b):
            # outside the bracket
            # switch to golden section
            return golden_section(a, b)
        current_step = abs(d - c)
        if(current_step > 0.5 * second_last_step):
            return golden_section(a, b)
        fd = f(d)
        fc = f(c)
        if(fd < fc):
            # d is in the new bracket
            if(d < c):
                # a d c
                b = c
                c = d
            else:
                # c d b
                a = c
                c = d
        else:
            # c is in the new bracket
            if(d < c):
                # d c b
                a = d
            else:
                # a c d
                b = d
    return c


def myplot(b_list, err_list):
    log_err = [np.log10(err) for err in err_list]
    fig, axs = plt.subplots(2,1, sharex=True)
    ax0, ax1 = axs[0], axs[1]
    #plot root
    ax0.scatter(range(len(b_list)), b_list, marker = 'o', facecolor = 'red', edgecolor = 'k')
    ax0.plot(range(len(b_list)), b_list, 'r-', alpha = .5)
    ax1.plot(range(len(err_list)), log_err,'.-')
    ax1.set_xlabel('number of iterations')
    ax0.set_ylabel(r'$x_{min}$')
    ax1.set_ylabel(r'$\log{\delta}$')
    plt.savefig('convergence.png')
    
if __name__ == "__main__":
    start_time = timeit.default_timer()
    myres = brent()
    mytime = timeit.default_timer() - start_time
    start_time = timeit.default_timer()
    scipyres = scipy.optimize.brent(f)
    scipytime = timeit.default_timer() - start_time
    print(f"my result: {myres}, time: {mytime}")
    print(f"scipy result: {scipyres}, time: {scipytime}")