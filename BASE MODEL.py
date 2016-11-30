import numpy as np
import matplotlib.pyplot as plt
import random
import time
import scipy.spatial

start = time.time()

#################### PARAMETERS #####################

# General
NR = 80  # initial number of retinal cells
NT = 80  # initial number of tectal cells
M = 7  # number of markers

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
deltat = 1  # time step
td = 5  # number of concentration iterations per weight iteration

# Synaptic modification
W = 1  # total strength available to each presynaptic fibre
h = 0.01  # ???
k = 0.03  # ???
elim = 0.005  # elimination threshold
Iterations = 10  # number of weight iterations

################### VARIABLES ###################
nR = NR  # present number of retinal cells (pre-surgery)
nT = NT  # present number of tectal cells (pre-surgery)

Wpt = np.zeros([nT, nR])  # synaptic strength between a presynaptic cell and a postsynaptic cell
Qpm = np.zeros([nR, M])  # presence of marker sources along retina
Qtm = np.zeros([nT, M])  # axonal flow of molecules into postsymaptic cells
Cpm = np.zeros([nR, M])  # concentration of a molecule in a presynaptic cell
Ctm = np.zeros([nT, M])  # concentration of a molecule in a postsynaptic cell
normalisedCpm = np.zeros([nR, M])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
normalisedCtm = np.zeros([nT, M])  # normalised (by marker conc.) marker concentration in a postsynaptic cell

################## RETINA #####################


# MARKER LOCATIONS
markerspacing = nR / (M - 1)
location = 0
for m in range(M - 1):
    Qpm[location, m] = Q
    location += markerspacing
Qpm[nR - 1, M - 1] = Q


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


averagemarkerchange = 1
while averagemarkerchange > stab:
    deltaconc = conc_change(Cpm, 'presynaptic')
    averagemarkerchange = sum(sum(deltaconc)) / sum(sum(Cpm)) * 100
    Cpm += (deltaconc * deltat)


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


normalisedCpm = normalise(Cpm)




#################### CONNECTIONS ######################


# INITIAL CONNECTIONS
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


initialconnections()


# ITERATIONS

def update_weight():
    # SYNAPTIC WEIGHT
    for p in range(nR):
        totalSp = 0
        connections = 0
        deltaWsum = np.zeros([nR])
        deltaWpt = np.zeros([nT, nR])
        Spt = np.zeros([nT, nR])
        meanSp = np.zeros([nR])

        for tectal in range(nT):
            # calculate similarity
            Spt[tectal, p] = 1 - scipy.spatial.distance.cosine(normalisedCpm[p, :], normalisedCtm[tectal, :])
            if Wpt[tectal, p] > 0:
                totalSp += Spt[tectal, p]
                connections += 1
        # calculate mean similarity
        meanSp[p] = (totalSp / connections) - k

        for tectal in range(nT):
            # calculate deltaW
            deltaWpt[tectal, p] = h * (Spt[tectal, p] - meanSp[p])
            # calculate deltaWsum
            if Wpt[tectal, p] > 0:
                deltaWsum[p] += deltaWpt[tectal, p]

        for tectal in range(nT):
            # calculate new W
            Wpt[tectal, p] = (Wpt[tectal, p] + deltaWpt[tectal, p]) * W / (W + deltaWsum[p])
            # REMOVE SYNAPSES
            if Wpt[tectal, p] < elim * W:
                Wpt[tectal, p] = 0

        # ADD NEW SYNAPSES
        if Wpt[0, p] == 0 and Wpt[1, p] > 0.02 * W:
            Wpt[0, p] = 0.01 * W
        if Wpt[nT - 1, p] == 0 and Wpt[nT - 2, p] > 0.02 * W:
            Wpt[nT - 1, p] = 0.01 * W
        for tectal in range(1, nT - 1):
            if Wpt[tectal, p] == 0 and (Wpt[tectal + 1, p] > 0.02 * W or Wpt[tectal - 1, p] > 0.02 * W):
                Wpt[tectal, p] = 0.01 * W


averagemarkerchange = 1
iterations = 0
while iterations < Iterations:
    Qtm = np.dot(Wpt, normalisedCpm)

    for t in range(td):
        deltaconc = conc_change(Ctm, 'tectal')
        #averagemarkerchange = sum(sum(deltaconc)) / sum(sum(Ctm)) * 100
        Ctm += (deltaconc * deltat)
    normalisedCtm = normalise(Ctm)
    update_weight()
    iterations += 1

##################### PLOT 1 #########################

params = {'font.size': '10'}
plt.rcParams.update(params)

plt.subplot(2, 1, 1)
for m in range(M):
    plt.plot(range(1, nT + 1), Ctm[0:nT, m])
plt.ylabel('Marker Concentration')
plt.xticks([], [])

plt.subplot(2, 1, 2)
plot3 = np.swapaxes(Wpt, 0, 1)
plt.pcolormesh(plot3, cmap='Greys')
plt.ylabel('Presynaptic Cell Number')
plt.xlabel('Postsynaptic Cell Number')

####################### END #########################

end = time.time()
elapsed = end - start
print('Time elapsed: ', elapsed, 'seconds')

plt.show()
