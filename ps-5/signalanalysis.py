import numpy as np
import matplotlib.pyplot as plt

# source: https://stackoverflow.com/questions/46473270/import-dat-file-as-an-array
def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False

data = []
with open('signal.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split('|')
        for i in k:
            if is_float(i):
                data.append(float(i)) 

data = np.array(data, dtype='float')
time_arr = data[::2]
signal_arr = data[1::2]
data_num = len(time_arr)

def qa():
    plt.scatter(time_arr, signal_arr, s=1)
    plt.title("Signal Data")
    plt.xlabel("time")
    plt.ylabel("signal")
    plt.savefig("signal.png")
    plt.clf()
    print(f"data length: {len(data_num)}")

def qb():
    '''
    fit a third-order polynomial using SVD
    '''
    # time_arr_T = time_arr.reshape(-1,1)
    # signal_arr_T = signal_arr.reshape(-1,1)
    T = np.ones((data_num, 4))
    T[:,1] = time_arr
    T[:,2] = pow(time_arr, 2)
    T[:,3] = pow(time_arr, 3)
    # print(T[:2])
    Y = signal_arr
    (u, w, vt) = np.linalg.svd(T, full_matrices=False)
    T_inv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
    A = T_inv.dot(Y)
    print(f"third-order polynomial fit: {A[0]} + {A[1]}t + {A[2]}t^2 + {A[3]}t^3")
    ys = T.dot(A)
    plt.scatter(time_arr, ys, s=1, label = "third-order polynomial fit")
    plt.scatter(time_arr, signal_arr, s=1, label = "original signal")
    plt.title("Third-order Polynomial Fit")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("signal")
    plt.savefig("third-order-poly.png")
    plt.clf()

def qb_time_scaled():
    '''
    fit a third-order polynomial using SVD
    '''
    time_arr_scaled = time_arr / 1e9
    T = np.ones((data_num, 4))
    T[:,1] = time_arr_scaled
    T[:,2] = pow(time_arr_scaled, 2)
    T[:,3] = pow(time_arr_scaled, 3)
    # print(T[:2])
    Y = signal_arr
    (u, w, vt) = np.linalg.svd(T, full_matrices=False)
    T_inv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
    A = T_inv.dot(Y)
    print(f"third-order polynomial fit (time scaled by 1e-9): {A[0]} + {A[1]}t + {A[2]}t^2 + {A[3]}t^3")
    ys = T.dot(A)
    plt.scatter(time_arr_scaled, ys, s=1, label = "third-order polynomial fit")
    plt.scatter(time_arr_scaled, signal_arr, s=1, label = "original signal")
    plt.title("Third-order Polynomial Fit (time scaled by 1e-9)")
    plt.legend()
    plt.xlabel("time (1e9*unit)")
    plt.ylabel("signal")
    plt.savefig("third-order-poly-time-scaled.png")
    plt.clf()

def qc():
    '''
    residual
    '''
    time_arr_scaled = time_arr / 1e9
    T = np.ones((data_num, 4))
    T[:,1] = time_arr_scaled
    T[:,2] = pow(time_arr_scaled, 2)
    T[:,3] = pow(time_arr_scaled, 3)
    Y = signal_arr
    (u, w, vt) = np.linalg.svd(T, full_matrices=False)
    T_inv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
    A = T_inv.dot(Y)
    ys = T.dot(A)
    residuals = Y - ys
    plt.scatter(time_arr_scaled, residuals, s = 1)
    plt.title("Residuals after Third-order Polynomial Fit")
    plt.xlabel("time (1e9*unit)")
    plt.ylabel("signal")
    plt.savefig("residual-third-order-poly.png")
    

# qa()
# qb()
# qb_time_scaled()
qc()