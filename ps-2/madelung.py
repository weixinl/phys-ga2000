# HW 2, Q 2
# Newman Excercise 2.9, P74
# the Mandelung constant

import numpy as np
import timeit
def madelung_for(_L):
    '''
    calculate madelung constant for electrical potential in a lattice
    '''
    val = 0
    for i in range(-_L, _L + 1):
        for j in range(-_L, _L + 1):
            for k in range(-_L, _L + 1):
                if(i == 0 and j == 0 and k == 0):
                    continue
                sign = 1
                if((i + j + k) % 2):
                    sign = -1
                val += sign * 1 / np.sqrt(i*i + j*j + k*k)
    
    return val

def potential(_i, _j, _k):
    if(_i == 0 and _j == 0 and _k == 0):
        return 0
    sign = 1
    if((_i + _j + _k) % 2):
        sign = -1
    return sign * 1 / np.sqrt(_i*_i + _j*_j + _k*_k)

def madelung_mesh(_L):
    '''
    calculate madelung constant for electrical potential in a lattice without for loop
    '''
    nums = range(-_L, _L + 1)
    xs, ys, zs = np.meshgrid(nums, nums, nums)
    potential_vec = np.vectorize(potential)
    potential_mat = potential_vec(xs, ys, zs)
    return np.sum(potential_mat)


L = 100
print(f"L: {L}")
const_for = None
const_mesh = None
start_time = timeit.default_timer()
const_for = madelung_for(L)
t_for = timeit.default_timer() - start_time
start_time = timeit.default_timer()
const_mesh = madelung_mesh(L)
t_mesh = timeit.default_timer() - start_time

print(f"for loop result: {const_for}, time: {t_for} seconds")
print(f"meshgrid result: {const_mesh}, time: {t_mesh} seconds")