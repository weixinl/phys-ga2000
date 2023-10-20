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
time_list = data[::2]
signal_list = data[1::2]

def qa():
    plt.scatter(time_list, signal_list, s=1)
    plt.title("Signal Data")
    plt.xlabel("time")
    plt.ylabel("signal")
    plt.savefig("signal.png")
    print(f"data length: {len(time_list)}")

def qb():
    '''
    fit a third-order polynomial
    '''

# qa()