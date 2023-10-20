import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fft import fft, fftfreq


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
    plt.clf()

def poly_fit(order = 4):
    '''
    fit the data using a polynomial of spacified order
    return the condition number which is the ratio between the maximum and minimun singular values
    '''
    time_arr_scaled = time_arr / 1e9
    T = np.ones((data_num, order + 1))
    # T[:,1] = time_arr_scaled
    # T[:,2] = pow(time_arr_scaled, 2)
    # T[:,3] = pow(time_arr_scaled, 3)
    for i in range(order + 1):
        T[:, i] = pow(time_arr_scaled, i)
    Y = signal_arr
    (u, w, vt) = np.linalg.svd(T, full_matrices=False)
    T_inv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
    A = T_inv.dot(Y)
    print(f"order {order} polynomial fit (time scaled by 1e-9): {A}")
    ys = T.dot(A)
    plt.scatter(time_arr_scaled, ys, s=1, label = f"order {order} polynomial fit")
    plt.scatter(time_arr_scaled, signal_arr, s=1, label = "original signal")
    plt.title(f"Order {order} Polynomial Fit (time scaled by 1e-9)")
    plt.legend()
    plt.xlabel("time (1e9*unit)")
    plt.ylabel("signal")
    plt.savefig(f"order-{order}-poly-time-scaled.png")
    plt.clf()
    return w[0]/w[order]

def qd():
    print(f"condition number: {poly_fit(19)}")

def qe():
    '''
    fit sin and cos functions plus a zero-point offset
    find period by fourier transformation from hint
    substract trend to get flat signal from hint
    '''
    time_arr_scaled = time_arr / 1e9
    #estimate linear trend, which is a first-order polynomial fit
    T = np.ones((data_num, 2))
    T[:,1] = time_arr_scaled
    Y = signal_arr
    (u, w, vt) = np.linalg.svd(T, full_matrices=False)
    T_inv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
    A = T_inv.dot(Y)
    ys = T.dot(A)
    plt.scatter(time_arr_scaled, ys, s=1, label = "linear fit")
    plt.scatter(time_arr_scaled, signal_arr, s=1, label = "original signal")
    plt.title("Linear Trend")
    plt.legend()
    plt.xlabel("time (1e9*unit)")
    plt.ylabel("signal")
    plt.savefig("linear-trend.png")
    plt.clf()

    # flat signal (residuals after linear fit)
    flat_signal = Y - ys
    plt.scatter(time_arr_scaled, flat_signal, s = 1)
    plt.title("Flat Signal")
    plt.xlabel("time (1e9*unit)")
    plt.ylabel("signal")
    plt.savefig("flat-signal.png")
    plt.clf()

    # get frequency by fourier transform
    data_arr = np.zeros((data_num, 2))
    data_arr[:,0] = time_arr_scaled
    data_arr[:,1] = flat_signal
    data_sorted = data_arr[data_arr[:, 0].argsort()] # sort by time
    T = (data_sorted[data_num - 1][0]-data_sorted[0][0])/999
    time_arr_interpolate = np.array(range(data_num))*T
    # f_interpolate = scipy.interpolate.interp1d(data_sorted[:,0], data_sorted[:, 1], kind = 5, fill_value="extrapolate")
    signal_arr_interpolate = np.interp(time_arr_interpolate, data_sorted[:,0], data_sorted[:, 1])
    # signal_arr_interpolate = f_interpolate(time_arr_interpolate)
    plt.scatter(time_arr_interpolate, signal_arr_interpolate, s=1)
    # plt.scatter(time_arr, signal_arr, s=1, label = "original signal")
    plt.title("Interpolation")
    # plt.legend()
    plt.xlabel("time (1e9*unit)")
    plt.ylabel("signal")
    plt.savefig("interpolation.png")
    plt.clf()
    yf = fft(signal_arr_interpolate)
    xf = fftfreq(data_num, T)[:data_num//2]
    # print("yf:")
    # print(yf)
    plt.scatter(xf, 2.0/data_num * np.abs(yf[0:data_num//2]), s=10)
    plt.grid()
    plt.xlim(0,10)
    omega = 2*np.pi*xf[np.argsort(2.0/data_num * np.abs(yf[0:data_num//2]))[::-1]][0]
    plt.vlines(omega/(2*np.pi), 0, 10, color = 'red')
    plt.title("Fourier Transform")
    plt.xlabel("omega/(2*pi)")
    plt.ylabel("f(omega)")
    plt.savefig("fourier.png")
    plt.clf()
    print(f"omega: {omega}")

    # fit on full signal
    T = np.ones((data_num, 4))
    T[:,1] = time_arr_scaled
    T[:,2] = np.cos(omega*time_arr_scaled)
    T[:,3] = np.sin(omega*time_arr_scaled)
    Y = signal_arr
    (u, w, vt) = np.linalg.svd(T, full_matrices=False)
    T_inv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
    A = T_inv.dot(Y)
    ys = T.dot(A)
    plt.scatter(time_arr, ys, s=1, label = "trigonometric fit")
    plt.scatter(time_arr, signal_arr, s=1, label = "original signal")
    plt.title("Trigonometric Fit")
    plt.legend()
    plt.xlabel("time (1e9*unit)")
    plt.ylabel("signal")
    plt.savefig("trigonometric.png")
    plt.clf()
    print(f"trigonometric fit (time scaled by 1e-9): {A[0]} + {A[1]}t + {A[2]}cos({omega}t) + {A[3]}sin({omega}t)")

    print(f"period: {2*np.pi/omega}")
    residuals = signal_arr - ys
    plt.scatter(time_arr_scaled, residuals, s = 1)
    plt.title("Residuals after Trigonometric Fit")
    plt.xlabel("time (1e9*unit)")
    plt.ylabel("residual signal")
    plt.savefig("residual-trigonometric.png")
    plt.clf()

# qa()
# qb()
# qb_time_scaled()
# qc()
# qd()
qe()