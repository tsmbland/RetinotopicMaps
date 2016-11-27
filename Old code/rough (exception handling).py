import numpy as np
import matplotlib.pyplot as plt
import random
import time
import scipy.spatial

start = time.time()

#################### PARAMETERS #####################

# General
NRdim1 = 10  # initial number of retinal cells
NRdim2 = 10

#NT = 80  # initial number of tectal cells
M = 5  # number of markers

# Presynaptic concentrations
a = 0.006  # (or 0.003) #decay constant
d = 0.3 # diffusion length constant
E = 0.01  # synaptic elimination threshold
Q = 100  # release of markers from source
stab = 0.1  # retinal stability threshold

# Establishment of initial contacts
n0 = 8  # number of initial random contact
NL = 60  # sets initial bias

# Tectal concentrations
deltat = 1  # time step
td = 5  # number of concentration iterations per weight iteration

# Synaptic modification
W = 1  # total strength available to each presynaptic fibre
h = 0.01  # ???
k = 0.03  # ???
elim = 0.005  # elimination threshold
Iterations = 20  # number of weight iterations

################### VARIABLES ###################
#nR = NR  # present number of retinal cells (pre-surgery)
#nT = NT  # present number of tectal cells (pre-surgery)

#Wpt = np.zeros([nT, nR])  # synaptic strength between a presynaptic cell and a postsynaptic cell
Qpm = np.zeros([M, NRdim1, NRdim2])  # presence of marker sources along retina
#Qtm = np.zeros([nT, M])  # axonal flow of molecules into postsymaptic cells
Cpm = np.zeros([M, NRdim1, NRdim2])  # concentration of a molecule in a presynaptic cell
#Ctm = np.zeros([nT, M])  # concentration of a molecule in a postsynaptic cell
#normalisedCpm = np.zeros([nR, M])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
#normalisedCtm = np.zeros([nT, M])  # normalised (by marker conc.) marker concentration in a postsynaptic cell

################## RETINA #####################



# MARKER LOCATIONS

Qpm[0, 0, 0] = Q
Qpm[1, 0, NRdim2-1] = Q
Qpm[2, NRdim1-1, 0] = Q
Qpm[3, NRdim1-1, NRdim2-1] = Q
Qpm[4, NRdim1/2, NRdim2/2] = Q






# PRESYNAPTIC CONCENTRATIONS

def conc_change(concmatrix, layer):
    if layer == 'presynaptic':
        Qmatrix = Qpm
    elif layer == 'tectal':
        Qmatrix = Qtm

    count = 0
    concchange = np.zeros([M, NRdim1, NRdim2])
    for m in range(M):
        for dim1 in range(NRdim1):
            for dim2 in range(NRdim2):
                try:
                    concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (concmatrix[m, dim1, dim2 + 1] + concmatrix[m, dim1, dim2 - 11] + concmatrix[m, dim1 + 1, dim2] + concmatrix[m, dim1 - 11, dim2] - 4 * concmatrix[m, dim1, dim2]) + Qmatrix[m, dim1, dim2])
                    count += 1
                except IndexError:
                    try:
                        concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (concmatrix[m, dim1 + 1, dim2] + concmatrix[m, dim1 - 1, dim2] + concmatrix[m, dim1, dim2 + 1] - 3 * concmatrix[m, dim1, dim2]) + Qmatrix[m, dim1, dim2])

                    except IndexError:
                        try:
                            concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (concmatrix[m, dim1 + 1, dim2] + concmatrix[m, dim1 - 1, dim2] + concmatrix[m, dim1, dim2 - 1] - 3 * concmatrix[m, dim1, dim2]) + Qmatrix[m, dim1, dim2])

                        except IndexError:
                            try:
                                concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (concmatrix[m, dim1, dim2 + 1] + concmatrix[m, dim1, dim2 - 1] + concmatrix[m, dim1 + 1, dim2] - 3 * concmatrix[m, dim1, dim2]) + Qmatrix[m, dim1, dim2])

                            except IndexError:
                                try:
                                    concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (concmatrix[m, dim1, dim2 + 1] + concmatrix[m, dim1, dim2 - 1] + concmatrix[m, dim1 - 1, dim2] - 3 * concmatrix[m, dim1, dim2]) + Qmatrix[m, dim1, dim2])

                                except IndexError:
                                    try:
                                        concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (concmatrix[m, dim1 + 1, dim2] + concmatrix[m, dim1, dim2 + 1] - 2 * concmatrix[m, dim1, dim2]) + Qmatrix[m, dim1, dim2])

                                    except IndexError:
                                        try:
                                            concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (concmatrix[m, dim1 - 1, dim2] + concmatrix[m, dim1, dim2 - 1] - 2 * concmatrix[m, dim1, dim2]) + Qmatrix[m, dim1, dim2])

                                        except IndexError:
                                            try:
                                                concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (concmatrix[m, dim1 + 1, dim2] + concmatrix[m, dim1, dim2 - 1] - 2 * concmatrix[m, dim1, dim2]) + Qmatrix[m, dim1, dim2])
                                            except IndexError:
                                                concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (concmatrix[m, dim1 - 1, dim2] + concmatrix[m, dim1, dim2 + 1] - 2 * concmatrix[m, dim1, dim2]) + Qmatrix[m, dim1, dim2])
    print(count)
    return concchange

for t in range(1):
    deltaconc = conc_change(Cpm, 'presynaptic')
    Cpm += (deltaconc * deltat)

### NOT WORKING