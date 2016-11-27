import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import random
import time

start = time.time()

###PARAMETER VALUES

# General
NR = 80  # number of retinal cells
NT = 80  # number of tectal cells
M = 7  # number of markers

# Presynaptic concentrations
a = 0.006  # (or 0.003) #decay constant
d = 0.3  # diffusion length constant
E = 0.01  # synaptic elimination threshold
Q = 100  # release of markers from source

# Establishment of initial contacts
n0 = 8  # number of initial random contact
NL = 8  # sets initial bias

# Tectal concentrations
deltat = 1  # time step
td = 5  # number of iterations???

# Synaptic modification
W = 1  # total strength available to each presynaptic fibre
h = 0.01  # ???
k = 0.03  # ???

###VARIABLES

Wpt = np.zeros([NR, NT])  # synaptic strength between a presynaptic cell and a postsynaptic cell
# deltaWpt = np.zeros([NR,NT]) #change in synaptic weight between a presynaptic cell and a postsynaptic cell

Qpm = np.zeros([NR, M])  # presence of marker sources along retina
Qtm = np.zeros([NT, M])  # postsymaptic Q matrix

Cpm = np.zeros([NR, M])  # concentration of a molecule in a presynaptic cell
# Ctm = np.zeros([NT,M]) #concentration of a molecule in a postsynaptic cell
# normalisedCpm = np.zeros([NR,M]) #normalised (by marker conc.) marker concentration  in a presynaptic cell
# normalisedCtm = np.zeros([NT,M]) #normalised (by marker conc.) marker concentration in a postsynaptic cell

# Spt = np.zeros([NR,NT]) #similarity between a presynaptic and a postsynaptic cell
# meanSp = np.zeros([NR]) #mean similarity for a presynaptic cell



###MARKER LOCATIONS
markerspacing = NR / (M - 1)
location = 0
for m in range(M - 1):
    Qpm[location, m] = Q
    location = location + markerspacing
Qpm[NR - 1, M - 1] = Q







###PRESYNAPTIC CONCENTRATIONS
def conc_change(concmatrix, m, layer):
    concarray = concmatrix[:, m]

    if layer == 'presynaptic':
        Qarray = Qpm[:, m]
    elif layer == 'tectal':
        Qarray = Qtm[:, m]

    length = len(concarray)
    conc_change = np.zeros([length])
    conc_change[0] = (-a * concarray[0] + d * (-concarray[0] + concarray[1]) + Qarray[0])
    conc_change[length - 1] = (
        -a * concarray[length - 1] + d * (concarray[length - 2] - concarray[length - 1]) + Qarray[length - 1])
    for cell in range(1, length - 1):
        conc_change[cell] = (
            -a * concarray[cell] + d * (concarray[cell - 1] - 2 * concarray[cell] + concarray[cell + 1]) + Qarray[cell])
    return conc_change


averagemarkerchange = 1
count = 0
while averagemarkerchange > 0.1:
    concchange = np.zeros([NR, M])
    percentconcchange = np.zeros([NR, M])
    totalpercentconcchange = np.zeros([M])

    for m in range(M):
        concchange[:, m] = conc_change(Cpm, m, 'presynaptic')
        totalconcchange = sum(concchange[:, m])
        totalpercentconcchange[m] = (totalconcchange / sum(Cpm[:, m])) * 100
        Cpm[:, m] = Cpm[:, m] + concchange[:, m]

    averagemarkerchange = totalpercentconcchange.mean()
    count = count + 1


###PLOT NORMALISED PRESYNAPTIC CONCENTRATIONS
for m in range(M):
    plt.plot(range(1, NR + 1), Cpm[:, m])
plt.xlabel('Presynaptic Cell Number')
plt.ylabel('Normalised Marker Concentration')
plt.show()

