import numpy as np
import matplotlib.pyplot as plt
import random
import time
import scipy.spatial

start = time.time()

#################### PARAMETERS #####################

# General
NRdim1 = 20  # initial number of retinal cells
NRdim2 = 20

# NT = 80  # initial number of tectal cells
M = 5  # number of markers

# Presynaptic concentrations
a = 0.006  # (or 0.003) #decay constant
d = 0.3  # diffusion length constant
E = 0.01  # synaptic elimination threshold
Q = 100  # release of markers from source
stab = 0.1  # retinal stability threshold

# Establishment of initial contacts
n0 = 8  # number of initial random contact
NL = 60  # sets initial bias

# Tectal concentrations
deltat = 0.1  # time step
td = 5  # number of concentration iterations per weight iteration

# Synaptic modification
W = 1  # total strength available to each presynaptic fibre
h = 0.01  # ???
k = 0.03  # ???
elim = 0.005  # elimination threshold
Iterations = 20  # number of weight iterations

################### VARIABLES ###################
# nR = NR  # present number of retinal cells (pre-surgery)
# nT = NT  # present number of tectal cells (pre-surgery)

rnm = np.zeros([NRdim1 + 2, NRdim2 + 2])  # retinal neuron map (fixed)
rnm[1:-1, 1:-1] = 1

# Wpt = np.zeros([nT, nR])  # synaptic strength between a presynaptic cell and a postsynaptic cell
Qpm = np.zeros([M, NRdim1 + 2, NRdim2 + 2])  # presence of marker sources along retina
# Qtm = np.zeros([nT, M])  # axonal flow of molecules into postsymaptic cells
Cpm = np.zeros([M, NRdim1 + 2, NRdim2 + 2])  # concentration of a molecule in a presynaptic cell
# Ctm = np.zeros([nT, M])  # concentration of a molecule in a postsynaptic cell
# normalisedCpm = np.zeros([nR, M])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
# normalisedCtm = np.zeros([nT, M])  # normalised (by marker conc.) marker concentration in a postsynaptic cell



################## RETINA #####################

# MARKER LOCATIONS

Qpm[0, 1, 1] = Q
Qpm[1, 1, NRdim2] = Q
Qpm[2, NRdim1, 1] = Q
Qpm[3, NRdim1, NRdim2] = Q
Qpm[4, NRdim1 / 2, NRdim2 / 2] = Q


# PRESYNAPTIC CONCENTRATIONS

def conc_change(concmatrix, layer):

    # Matrix size
    Ndim1 = len(concmatrix[0, :, 0]) - 2
    Ndim2 = len(concmatrix[0, 0, :]) - 2

    # Neuron map
    nm = np.zeros([Ndim1 + 2, Ndim2 + 2])
    nm[1:-1, 1:-1] = 1

    # Neighbour Count
    nc = np.zeros([Ndim1 + 2, Ndim2 + 2])
    for dim1 in range(1, Ndim1 + 1):
        for dim2 in range(1, Ndim2 + 1):
            nc[dim1, dim2] = nm[dim1 + 1, dim2] + nm[dim1 - 1, dim2] + nm[dim1, dim2 + 1] + nm[dim1, dim2 - 1]

    # Qmatrix
    if layer == 'presynaptic':
        Qmatrix = Qpm
    elif layer == 'tectal':
        Qmatrix = Qtm

    # Conc change
    concchange = np.zeros([M, Ndim1 + 2, Ndim2 + 2])
    for m in range(M):
        for dim1 in range(1, Ndim1 + 1):
            for dim2 in range(1, Ndim2 + 1):
                concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (
                    concmatrix[m, dim1, dim2 + 1] + concmatrix[m, dim1, dim2 - 1] + concmatrix[m, dim1 + 1, dim2] +
                    concmatrix[m, dim1 - 1, dim2] - nc[dim1, dim2] * concmatrix[m, dim1, dim2]) + Qmatrix[m, dim1, dim2])

    return concchange



averagemarkerchange = 1
while averagemarkerchange > stab:
    deltaconc = conc_change(Cpm, 'presynaptic')
    averagemarkerchange = sum(sum(sum(deltaconc))) / sum(sum(sum(Cpm))) * 100
    Cpm += (deltaconc * deltat)

##################### PLOT #######################
params = {'font.size': '10'}
plt.rcParams.update(params)

for m in range(1, M + 1):
    plt.subplot(3, 2, m)
    plt.pcolormesh(Cpm[m - 1, 1:NRdim1+1, 1:NRdim2+1])
    plt.ylabel('Dimension 1')
    plt.xlabel('Dimension 2')

###################### END ########################
end = time.time()
elapsed = end - start
print('Time elapsed: ', elapsed, 'seconds')

plt.show()