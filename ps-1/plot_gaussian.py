import numpy as np
import matplotlib.pyplot as plt

def gaussian_prob(_x, _mean = 0, _sigma = 3):
    '''
    probability density function of Gaussian Distribution
    '''
    return 1 / (_sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * pow((_x - _mean)/_sigma,2))


xs = np.arange(-10, 10, 0.2, dtype = float)
ys = np.empty_like(xs)
num = len(xs)
for i in range(num):
    ys[i] = gaussian_prob(xs[i])
plt.plot(xs, ys)
plt.xlabel("x")
plt.ylabel("p (Probability Density)")
plt.title("Gaussian Distribution")
plt.savefig("gaussian.png")