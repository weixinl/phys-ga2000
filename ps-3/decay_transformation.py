# HW 3 Q 4
# Newman Excercise 10.4, P 460
# Radioactive decay again

import numpy as np
import matplotlib.pyplot as plt

tau = 3.053 * 60
mu = np.log(2)/tau

def transform(_z):
    '''
    input: a random number from [0, 1)
    output: a number from nonuniform distribution
    '''
    return -1 / mu * np.log(1 - _z)

N = 1000
randoms = np.random.rand(N)
transform_vec = np.vectorize(transform)
# chosen decay time
ts = transform_vec(randoms)
ts_sorted = np.sort(ts)
numbers = np.flip(np.array(range(1000)))
plt.plot(ts_sorted, numbers)
plt.title("Decay of Tl208 Using Transformation")
plt.xlabel("x (s)")
plt.ylabel("number remaining")
plt.savefig("decay-transform.png")