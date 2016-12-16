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
Iterations = 500  # number of weight iterations

################### VARIABLES ###################
nR = NR  # present number of retinal cells (pre-surgery)
nT = NT  # present number of tectal cells (pre-surgery)

Wpt = np.zeros([nT+2, nR+2])  # synaptic strength between a presynaptic cell and a postsynaptic cell
Qpm = np.zeros([nR+2, M])  # presence of marker sources along retina
Qtm = np.zeros([nT+2, M])  # axonal flow of molecules into postsymaptic cells
Cpm = np.zeros([nR+2, M])  # concentration of a molecule in a presynaptic cell
Ctm = np.zeros([nT+2, M])  # concentration of a molecule in a postsynaptic cell
normalisedCpm = np.zeros([nR+2, M])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
normalisedCtm = np.zeros([nT+2, M])  # normalised (by marker conc.) marker concentration in a postsynaptic cell

################## RETINA #####################


# MARKER LOCATIONS
markerspacing = nR / (M - 1)
location = 1
for m in range(M - 1):
    Qpm[location, m] = Q
    location += markerspacing
Qpm[nR, M - 1] = Q


# PRESYNAPTIC CONCENTRATIONS
def conc_change(concmatrix, layer):

    # Matrix size
    length = len(concmatrix[:, 0]) - 2

    # Neuron map
    nm = np.zeros([length + 2])
    nm[1:-1] = 1

    # Neighbour Count
    nc = np.zeros([length + 2])
    for cell in range(1, length + 1):
            nc[cell] = nm[cell-1] + nm[cell+1]

    # Qmatrix
    if layer == 'presynaptic':
        Qmatrix = Qpm
    elif layer == 'tectal':
        Qmatrix = Qtm

    # Conc change
    concchange = np.zeros([length+2, M])
    for m in range(M):
        for cell in range(1, length + 1):
            concchange[cell, m] = (-a * concmatrix[cell, m] + d * (
                concmatrix[cell - 1, m] - nc[cell] * concmatrix[cell, m] + concmatrix[cell + 1, m]) + Qmatrix[cell, m])

    return concchange


averagemarkerchange = 1
while averagemarkerchange > stab:
    deltaconc = conc_change(Cpm, 'presynaptic')
    averagemarkerchange = sum(sum(deltaconc)) / sum(sum(Cpm)) * 100
    Cpm += (deltaconc * deltat)


# NORMALISED PRESYNAPTIC CONCENTRATIONS
def normalise(concmatrix):
    length = len(concmatrix[:, 0]) - 2
    normalised = np.zeros([length + 2, M])
    markersum = np.zeros([length + 2])
    for cell in range(1, length + 1):
        markersum[cell] = sum(concmatrix[cell, :])

    for m in range(M):
        for cell in range(1, length + 1):
            normalised[cell, m] = concmatrix[cell, m] / markersum[cell]
            if normalised[cell, m] < E:
                normalised[cell, m] = 0

    return normalised


normalisedCpm = normalise(Cpm)



#################### CONNECTIONS ######################


# INITIAL CONNECTIONS
def initialconnections():
    initialstrength = W / n0
    arrangement = np.zeros([NL])
    arrangement[0:n0] = initialstrength

    for p in range(1,nR+1):
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
    for p in range(1, nR+1):
        totalSp = 0
        connections = 0
        deltaWsum = np.zeros([nR+2])
        deltaWpt = np.zeros([nT+2, nR+2])
        Spt = np.zeros([nT+2, nR+2])
        meanSp = np.zeros([nR+2])

        for tectal in range(1, nT+1):

            # Calculate similarity
                #Spt[tectal, p] = 1 - scipy.spatial.distance.cosine(normalisedCpm[p, :], normalisedCtm[tectal, :])
            for m in range(M):
                Spt[tectal, p] += min(normalisedCpm[p, m], normalisedCtm[tectal, m])

            # Count connections
            if Wpt[tectal, p] > 0:
                totalSp += Spt[tectal, p]
                connections += 1

        # Calculate mean similarity
        meanSp[p] = (totalSp / connections) - k

        for tectal in range(1, nT+1):

            # Calculate deltaW
            deltaWpt[tectal, p] = h * (Spt[tectal, p] - meanSp[p])

            # Calculate deltaWsum
            if Wpt[tectal, p] > 0:
                deltaWsum[p] += deltaWpt[tectal, p]

        for tectal in range(1, nT+1):

            # Calculate new W
            Wpt[tectal, p] = (Wpt[tectal, p] + deltaWpt[tectal, p]) * W / (W + deltaWsum[p])

            # REMOVE SYNAPSES
            if Wpt[tectal, p] < elim * W:
                Wpt[tectal, p] = 0

        # ADD NEW SYNAPSES
        for tectal in range(1, nT + 1):
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
    plt.plot(range(1, nT + 1), Ctm[1:nT+1, m])
plt.ylabel('Marker Concentration')
plt.xticks([], [])

plt.subplot(2, 1, 2)
plot3 = np.swapaxes(Wpt, 0, 1)
plot3 = plot3[1:nT+1, 1:nR+1]
plt.pcolormesh(plot3, cmap='Greys')
plt.ylabel('Presynaptic Cell Number')
plt.xlabel('Postsynaptic Cell Number')

####################### END #########################

end = time.time()
elapsed = end - start
print('Time elapsed: ', elapsed, 'seconds')

plt.show()