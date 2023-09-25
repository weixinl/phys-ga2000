# HW3 Q2
# Newman Example 4.3 P 137 Matrix Multiplication

import numpy as np
import timeit
import matplotlib.pyplot as plt

def matmul_loop(_A, _B):
    '''
    matrix multiplication by loop
    A: m*n, B: n*t
    C = A*B
    C: m*t
    '''
    m = _A.shape[0]
    n = _A.shape[1]
    t = _B.shape[1]
    C = np.zeros([m, t])
    for i in range(m):
        for j in range(t):
            for k in range(n):
                C[i, j] += _A[i, k] * _B[k, j]
    return C

def matmul_np(_A, _B):
    '''
    matrix multiplication by np.dot
    '''
    return np.dot(_A, _B)

N_list = list(range(20, 150, 10))
t_loop_list = []
t_np_list = []
for N in N_list:
    X = np.random.rand(N,N)
    start_time = timeit.default_timer()
    C = matmul_loop(X, X)
    t_loop = timeit.default_timer() - start_time
    t_loop_list.append(t_loop)
    # print(f"multiplication time of {N}*{N} matrix and {N}*{N} matrix by loop: {t_loop} seconds")
    start_time = timeit.default_timer()
    C = matmul_np(X, X)
    t_np = timeit.default_timer() - start_time
    t_np_list.append(t_np)
    # print(f"multiplication time of {N}*{N} matrix and {N}*{N} matrix by numpy.dot: {t_np} seconds")

# if t(N) = N^3, then log t = 3 log N
log_N_list = np.log(np.array(N_list))
log_t_loop_list = np.log(np.array(t_loop_list))
log_t_np_list = np.log(np.array(t_np_list))
plt.plot(log_N_list, log_t_loop_list)
plt.xlabel("log(N)")
plt.ylabel("log(t)")
plt.title("Matrix multiplication by loop")
plt.savefig("matmult-loop.png")
arr_1d_poly = np.polyfit(log_N_list, log_t_loop_list, 1)
print(f"by loop: log(t) = {arr_1d_poly[0]} * log(N) + ({arr_1d_poly[1]})")
plt.clf() # clean previous plt image
plt.plot(log_N_list, log_t_np_list)
plt.xlabel("log(N)")
plt.ylabel("log(t)")
plt.title("Matrix multiplication by np.dot")
plt.savefig("matmult-np.png")
arr_1d_np = np.polyfit(log_N_list, log_t_loop_list, 1)
print(f"by np.dot: log(t) = {arr_1d_np[0]} * log(N) + ({arr_1d_np[1]})")