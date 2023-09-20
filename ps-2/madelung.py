import numpy as np
from timeit import timeit
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


L = 50
print(f"L: {L}")
print(f"result by for loop: {madelung_for(L)}")
print(f"result by meshgrid: {madelung_mesh(L)}")