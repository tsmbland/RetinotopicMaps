import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys

start = time.time()

#################### PARAMETERS #####################

# General
Iterations = 100  # number of weight iterations
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

###############  VARIABLES ###################
rmin = 1
rmax = NR
tmin = 1
tmax = NT
nR = rmax - rmin + 1  # present number of retinal cells (pre-surgery)
nT = tmax - tmin + 1  # present number of tectal cells (pre-surgery)

Wpt = np.zeros([Iterations + 1, NT + 2, NR + 2])  # synaptic strength between a presynaptic cell and a postsynaptic cell
Qpm = np.zeros([NR + 2, M])  # presence of marker sources along retina
Qtm = np.zeros([NT + 2, M])  # axonal flow of molecules into postsymaptic cells
Cpm = np.zeros([NR + 2, M])  # concentration of a molecule in a presynaptic cell
Ctm = np.zeros([NT + 2, M])  # concentration of a molecule in a postsynaptic cell
normalisedCpm = np.zeros([NR + 2, M])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
normalisedCtm = np.zeros([NT + 2, M])  # normalised (by marker conc.) marker concentration in a postsynaptic cell

currentiteration = 0


################# FUNCTIONS #####################

def conc_change(concmatrix, layer):
    # Layer
    if layer == 'presynaptic':
        Qmatrix = Qpm
        start = rmin
        end = rmax
    elif layer == 'tectal':
        Qmatrix = Qtm
        start = tmin
        end = tmax

    # Matrix size
    length = len(concmatrix[:, 0])

    # Neuron map
    nm = np.zeros([length])
    nm[start:end + 1] = 1

    # Neighbour Count
    nc = np.zeros([length])
    for cell in range(start, end + 1):
        nc[cell] = nm[cell - 1] + nm[cell + 1]

    # Conc change
    concchange = np.zeros([length, M])
    for m in range(M):
        for cell in range(start, end + 1):
            concchange[cell, m] = (-a * concmatrix[cell, m] + d * (
                concmatrix[cell - 1, m] - nc[cell] * concmatrix[cell, m] + concmatrix[cell + 1, m]) + Qmatrix[cell, m])

    return concchange


def normalise(concmatrix, layer):
    # Layer
    if layer == 'presynaptic':
        start = rmin
        end = rmax
    elif layer == 'tectal':
        start = tmin
        end = tmax

    # Matrix size
    length = len(concmatrix[:, 0])

    # Normalisation
    normalised = np.zeros([length, M])
    markersum = np.zeros([length])
    for cell in range(start, end + 1):
        markersum[cell] = sum(concmatrix[cell, :])

    for m in range(M):
        for cell in range(start, end + 1):
            normalised[cell, m] = concmatrix[cell, m] / markersum[cell]
            if normalised[cell, m] < E:
                normalised[cell, m] = 0

    return normalised


def initialconnections(p):
    initialstrength = W / n0
    arrangement = np.zeros([NL])
    arrangement[0:n0] = initialstrength

    if int(p * ((NT - NL) / NR) + NL) <= tmax:
        random.shuffle(arrangement)
        Wpt[0, int(p * ((NT - NL) / NR)) + 1: int(p * ((NT - NL) / NR) + NL) + 1, p] = arrangement
    else:
        shrunkarrangement = np.zeros([tmax - int(p * ((NT - NL) / NR))])
        shrunkarrangement[0:n0] = initialstrength
        random.shuffle(shrunkarrangement)
        Wpt[0, int(p * ((NT - NL) / NR)) + 1: tmax + 1, p] = shrunkarrangement


def updateWeight():
    # SYNAPTIC WEIGHT
    totalSp = np.zeros([NR + 2])
    connections = np.zeros([NR + 2])
    deltaWsum = np.zeros([NR + 2])
    deltaWpt = np.zeros([NT + 2, NR + 2])
    Spt = np.zeros([NT + 2, NR + 2])

    for p in range(rmin, rmax + 1):
        for tectal in range(tmin, tmax + 1):

            # Calculate similarity
            for m in range(M):
                Spt[tectal, p] += min(normalisedCpm[p, m], normalisedCtm[tectal, m])

            # Count connections
            if Wpt[currentiteration - 1, tectal, p] > 0:
                totalSp[p] += Spt[tectal, p]
                connections[p] += 1

        # Calculate mean similarity
        meanSp = (totalSp[p] / connections[p]) - k

        for tectal in range(tmin, tmax + 1):

            # Calculate deltaW
            deltaWpt[tectal, p] = h * (Spt[tectal, p] - meanSp)

            # Calculate deltaWsum
            if Wpt[currentiteration - 1, tectal, p] > 0:
                deltaWsum[p] += deltaWpt[tectal, p]

        for tectal in range(tmin, tmax + 1):
            # Calculate new W
            Wpt[currentiteration, tectal, p] = (Wpt[currentiteration - 1, tectal, p] + deltaWpt[tectal, p]) * W / (
                W + deltaWsum[p])


def removesynapses():
    for p in range(rmin, rmax + 1):
        for tectal in range(tmin, tmax + 1):
            if Wpt[currentiteration, tectal, p] < elim * W:
                Wpt[currentiteration, tectal, p] = 0


def addsynapses():
    for p in range(rmin, rmax + 1):
        for tectal in range(tmin, tmax + 1):
            if Wpt[currentiteration, tectal, p] == 0 and (
                            Wpt[currentiteration, tectal + 1, p] > 0.02 * W or Wpt[
                        currentiteration, tectal - 1, p] > 0.02 * W):
                Wpt[currentiteration, tectal, p] = 0.01 * W


################## ALGORITHM #####################

# MARKER LOCATIONS
markerspacing = NR / (M - 1)
location = 1
for m in range(M - 1):
    Qpm[location, m] = Q
    location += markerspacing
Qpm[NR, M - 1] = Q

# PRESYNAPTIC CONCENTRATIONS
averagemarkerchange = 1
while averagemarkerchange > stab:
    deltaconc = conc_change(Cpm, 'presynaptic')
    averagemarkerchange = (sum(sum(deltaconc)) / sum(sum(Cpm))) * 100
    Cpm += (deltaconc * deltat)

# NORMALISED PRESYNAPTIC CONCENTRATIONS
normalisedCpm = normalise(Cpm, 'presynaptic')

# INITIAL CONNECTIONS
for p in range(rmin, rmax + 1):
    initialconnections(p)

# INITIAL CONCENTRATIONS
Qtm = np.dot(Wpt[currentiteration, :, :], normalisedCpm)
for t in range(td):
    deltaconc = conc_change(Ctm, 'tectal')
    Ctm += (deltaconc * deltat)
normalisedCtm = normalise(Ctm, 'tectal')

# ITERATIONS
for iteration in range(Iterations):
    currentiteration += 1
    updateWeight()
    removesynapses()
    addsynapses()

    Qtm = np.dot(Wpt[currentiteration, :, :], normalisedCpm)
    for t in range(td):
        deltaconc = conc_change(Ctm, 'tectal')
        Ctm += (deltaconc * deltat)
    normalisedCtm = normalise(Ctm, 'tectal')

    sys.stdout.write('\r%i percent' % (iteration * 100 / Iterations))
    sys.stdout.flush()

##################### PLOT ####################

plt.subplot2grid((int(1.5 * NR), NT), (0, 0), rowspan=int(0.5 * NR), colspan=NT)
for m in range(M):
    plt.plot(range(tmin, tmax + 1), Ctm[tmin:tmax + 1, m])
plt.ylabel('Marker Concentration')
plt.xticks([], [])


def tabulate_weight_matrix():
    table = np.zeros([nR * nT, 4])
    row = 0
    deltaw = Wpt[currentiteration, :, :] - Wpt[currentiteration - 1, :, :]
    for p in range(rmin, rmax + 1):
        for tectal in range(tmin, tmax + 1):
            table[row, 0] = p
            table[row, 1] = tectal
            table[row, 2] = Wpt[currentiteration - 1, tectal, p]
            if deltaw[tectal, p] >= 0:
                table[row, 3] = 1
            else:
                table[row, 3] = 0
            row += 1
    return table


plt.subplot2grid((int(1.5 * NR), NT), (int(0.5 * NR), 0), rowspan=NR, colspan=NT)
plot = tabulate_weight_matrix()
plt.scatter(plot[:, 1], plot[:, 0], s=(plot[:, 2]) * 20, marker='s', c=(plot[:, 3]), cmap='Greys', edgecolors='k')
plt.clim(0, 1)
plt.ylabel('Presynaptic Cell Number')
plt.xlabel('Postsynaptic Cell Number')
plt.xlim([tmin - 1, tmax])
plt.ylim([rmin - 1, rmax])

####################### END #########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('Time elapsed: ', elapsed, 'seconds')

params = {'font.size': '10'}
plt.rcParams.update(params)
plt.tight_layout()
plt.show()
