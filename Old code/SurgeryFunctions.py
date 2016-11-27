import numpy as np
import matplotlib.pyplot as plt
import random
import time
import scipy.spatial





#################### PARAMETERS #####################

# General
NR = 80  # initial number of retinal cells
NT = 80  # initial number of tectal cells
nR = 80  # present number of retinal cells (pre-surgery)
nT = 80  # present number of tectal cells (pre-surgery)
M = 7  # number of markers

# Presynaptic concentrations
a = 0.006  # (or 0.003) #decay constant
d = 0.3  # diffusion length constant
E = 0.01  # synaptic elimination threshold
Q = 100  # release of markers from source
stab = 0.1  # stability threshold

# Establishment of initial contacts
n0 = 8  # number of initial random contact
NL = 60  # sets initial bias

# Tectal concentrations
deltat = 1  # time step
td = 5  # number of iterations???

# Synaptic modification
W = 1  # total strength available to each presynaptic fibre
h = 0.01  # ???
k = 0.03  # ???
elim = 0.005  # elimination threshold




################### VARIABLES ###################

Wpt = np.zeros([nT, nR])  # synaptic strength between a presynaptic cell and a postsynaptic cell
deltaWpt = np.zeros([nT, nR])  # change in synaptic weight between a presynaptic cell and a postsynaptic cell
deltaWsum = np.zeros([nR])

Qpm = np.zeros([nR, M])  # presence of marker sources along retina
Qtm = np.zeros([nT, M])  # axonal flow of molecules into postsymaptic cells

Cpm = np.zeros([nR, M])  # concentration of a molecule in a presynaptic cell
Ctm = np.zeros([nT, M])  # concentration of a molecule in a postsynaptic cell
normalisedCpm = np.zeros([nR, M])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
normalisedCtm = np.zeros([nT, M])  # normalised (by marker conc.) marker concentration in a postsynaptic cell

Spt = np.zeros([nT, nR])  # similarity between a presynaptic and a postsynaptic cell
meanSp = np.zeros([nR])  # mean similarity for a presynaptic cell


# PRESYNAPTIC CONCENTRATIONS
def conc_change(concmatrix, layer):
    if layer == 'presynaptic':
        Qmatrix = Qpm
    elif layer == 'tectal':
        Qmatrix = Qtm

    length = len(concmatrix[:, 0])
    concchange = np.zeros([length, M])
    for m in range(M):
        concchange[0, m] = (-a * concmatrix[0, m] + d * (-concmatrix[0, m] + concmatrix[1, m]) + Qmatrix[0, m])
        concchange[length - 1, m] = (
            -a * concmatrix[length - 1, m] + d * (concmatrix[length - 2, m] - concmatrix[length - 1, m]) + Qmatrix[
                length - 1, m])
        for cell in range(1, length - 1):
            concchange[cell, m] = (-a * concmatrix[cell, m] + d * (
                concmatrix[cell - 1, m] - 2 * concmatrix[cell, m] + concmatrix[cell + 1, m]) + Qmatrix[cell, m])
    return concchange



# NORMALISED PRESYNAPTIC CONCENTRATIONS
def normalise(concmatrix):
    length = len(concmatrix[:, 0])
    normalised = np.zeros([length, M])
    markersum = np.zeros([length])
    for p in range(length):
        markersum[p] = sum(concmatrix[p, :])

    for m in range(M):
        for p in range(length):
            normalised[p, m] = concmatrix[p, m] / markersum[p]
            if normalised[p, m] < E:
                normalised[p, m] = 0

    return normalised



def initialconnections():
    initialstrength = W / n0
    arrangement = np.zeros([NL])
    arrangement[0:n0] = initialstrength

    for p in range(nR):
        if np.ceil(p * ((NT - NL) / NR) + NL) < nT:
            random.shuffle(arrangement)
            Wpt[np.ceil(p * ((NT - NL) / NR)): np.ceil(p * ((NT - NL) / NR) + NL), p] = arrangement
        else:
            shrunkarrangement = np.zeros([nT - np.ceil(p * ((NT - NL) / NR))])
            shrunkarrangement[0:n0] = initialstrength
            random.shuffle(shrunkarrangement)
            Wpt[np.ceil(p * ((NT - NL) / NR)): nT, p] = shrunkarrangement

