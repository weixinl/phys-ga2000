# HW2 Q3
# Newman Excercise 3.7 P122
import numpy as np
import matplotlib.pyplot as plt

def is_mandelbrot(_x, _y):
    '''
    check if the input is in mandelbrot set
    '''
    c = complex(_x, _y)
    iternum = 1000
    z = 0
    for i in range(iternum):
        z = z * z + c
        if(abs(z) > 2):
            return 0
    return 1

N = 500 # N*N grid
step_size = 4.0/N
coords = np.arange(-2, 2, step_size)
# print(coords)
xs, ys = np.meshgrid(coords, coords)
# print(xs)
is_mandelbrot_vec = np.vectorize(is_mandelbrot)
res_mat = is_mandelbrot_vec(xs, ys)
# print(res_mat)
res_is, res_js = np.where(res_mat == 1)
res_xs = -2 + step_size * res_js
res_ys = -2 + step_size * res_is
plt.scatter(res_xs, res_ys)
plt.xlabel("Re")
plt.ylabel("Im")
plt.title("Mandelbrot Set")
plt.savefig("mandelbrot.png")

    

