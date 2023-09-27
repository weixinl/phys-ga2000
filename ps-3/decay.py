# HW 3 Q 3
# Newman Exercise 10.2, P 456
# Radioactive Decay Chain

import numpy as np
import matplotlib.pyplot as plt

def calc_prob(_tau):
    '''
    output: probability of decay in 1 s
    input: half-life (s)
    '''
    return 1 - pow(2, - 1/ _tau)

tauBi = 46*60
tauTl = 2.2*60
tauPb = 3.3*60

# probability of decay in 1 s
pBi = calc_prob(tauBi)
pTl = calc_prob(tauTl)
pPb = calc_prob(tauPb)

NBi213 = 10000
NTl = 0
NPb = 0
NBi209 = 0

NBi213_list = []
NTl_list = []
NPb_list = []
NBi209_list = []
t = 20000
t_list = list(range(t))
for i in range(t):
    NBi213_list.append(NBi213)
    NTl_list.append(NTl)
    NPb_list.append(NPb)
    NBi209_list.append(NBi209)
    n = NPb
    for j in range(n):
        if np.random.random() < pPb:
            NBi209 += 1
            NPb -= 1
    n = NTl
    for j in range(n):
        if np.random.random() < pTl:
            NPb += 1
            NTl -= 1
    n = NBi213
    for j in range(n):
        if np.random.random() < pBi:
            # decay
            if np.random.random() < 0.9791:
                NPb += 1
            else:
                NTl += 1
            NBi213 -= 1
assert(NBi213 + NTl + NPb + NBi209 == 10000)
plt.plot(t_list, NBi213_list, label = 'Bi213')
plt.plot(t_list, NTl_list, label = 'Tl209')
plt.plot(t_list, NPb_list, label = 'Pb209')
plt.plot(t_list, NBi209_list, label = 'Bi209')
plt.title("Radioactive Decay Chain")
plt.legend()
plt.xlabel("t (s)")
plt.ylabel("number")
plt.savefig("decay.png")
