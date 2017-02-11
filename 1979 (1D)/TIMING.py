import numpy as np
import matplotlib.pyplot as plt
import random
import time

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
Iterations = 2000  # number of weight iterations


###############  VARIABLES ###################
rmin = 1
rmax = NR
tmin = 1
tmax = NT
nR = rmax - rmin + 1  # present number of retinal cells (pre-surgery)
nT = tmax - tmin + 1  # present number of tectal cells (pre-surgery)

Wpt = np.zeros([NT+2, NR+2])  # synaptic strength between a presynaptic cell and a postsynaptic cell
Qpm = np.zeros([NR+2, M])  # presence of marker sources along retina
Qtm = np.zeros([NT+2, M])  # axonal flow of molecules into postsymaptic cells
Cpm = np.zeros([NR+2, M])  # concentration of a molecule in a presynaptic cell
Ctm = np.zeros([NT+2, M])  # concentration of a molecule in a postsynaptic cell
normalisedCpm = np.zeros([NR+2, M])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
normalisedCtm = np.zeros([NT+2, M])  # normalised (by marker conc.) marker concentration in a postsynaptic cell

################## RETINA #####################


# MARKER LOCATIONS
markerspacing = NR / (M - 1)
location = 1
for m in range(M - 1):
    Qpm[location, m] = Q
    location += markerspacing
Qpm[NR, M - 1] = Q


# PRESYNAPTIC CONCENTRATIONS
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
            nc[cell] = nm[cell-1] + nm[cell+1]

    # Conc change
    concchange = np.zeros([length, M])
    for m in range(M):
        for cell in range(start, end + 1):
            concchange[cell, m] = (-a * concmatrix[cell, m] + d * (
                concmatrix[cell - 1, m] - nc[cell] * concmatrix[cell, m] + concmatrix[cell + 1, m]) + Qmatrix[cell, m])

    return concchange


averagemarkerchange = 1
while averagemarkerchange > stab:
    deltaconc = conc_change(Cpm, 'presynaptic')
    averagemarkerchange = sum(sum(deltaconc)) / sum(sum(Cpm)) * 100
    Cpm += (deltaconc * deltat)


# NORMALISED PRESYNAPTIC CONCENTRATIONS
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


normalisedCpm = normalise(Cpm, 'presynaptic')



#################### CONNECTIONS ######################


# INITIAL CONNECTIONS
def initialconnections(p):
    initialstrength = W / n0
    arrangement = np.zeros([NL])
    arrangement[0:n0] = initialstrength

    if int(p * ((NT - NL) / NR) + NL) <= tmax:
        random.shuffle(arrangement)
        Wpt[int(p * ((NT - NL) / NR)) + 1: int(p * ((NT - NL) / NR) + NL) + 1, p] = arrangement
    else:
        shrunkarrangement = np.zeros([tmax - int(p * ((NT - NL) / NR))])
        shrunkarrangement[0:n0] = initialstrength
        random.shuffle(shrunkarrangement)
        Wpt[int(p * ((NT - NL) / NR)) + 1: tmax + 1, p] = shrunkarrangement


for p in range(rmin, rmax + 1):
    initialconnections(p)


# INITIAL CONCENTRATIONS
Qtm = np.dot(Wpt, normalisedCpm)
for t in range(td):
    deltaconc = conc_change(Ctm, 'tectal')
    Ctm += (deltaconc * deltat)
normalisedCtm = normalise(Ctm, 'tectal')


# ITERATIONS

def weight_change():
    # SYNAPTIC WEIGHT

    newweight = np.zeros([NT + 2, NR + 2])
    for p in range(rmin, rmax + 1):
        totalSp = 0
        connections = 0
        deltaWsum = np.zeros([NR + 2])
        deltaWpt = np.zeros([NT + 2, NR + 2])
        Spt = np.zeros([NT + 2, NR + 2])
        meanSp = np.zeros([NR + 2])

        for tectal in range(tmin, tmax + 1):

            # Calculate similarity
            for m in range(M):
                Spt[tectal, p] += min(normalisedCpm[p, m], normalisedCtm[tectal, m])

            # Count connections
            if Wpt[tectal, p] > 0:
                totalSp += Spt[tectal, p]
                connections += 1

        # Calculate mean similarity
        meanSp[p] = (totalSp / connections) - k

        for tectal in range(tmin, tmax + 1):

            # Calculate deltaW
            deltaWpt[tectal, p] = h * (Spt[tectal, p] - meanSp[p])

            # Calculate deltaWsum
            if Wpt[tectal, p] > 0:
                deltaWsum[p] += deltaWpt[tectal, p]

        for tectal in range(tmin, tmax + 1):

            # Calculate new W
            newweight[tectal, p] = (Wpt[tectal, p] + deltaWpt[tectal, p]) * W / (W + deltaWsum[p])

            # REMOVE SYNAPSES
            if Wpt[tectal, p] < elim * W:
                newweight[tectal, p] = 0

        # ADD NEW SYNAPSES
        for tectal in range(tmin, tmax + 1):
            if Wpt[tectal, p] == 0 and (Wpt[tectal + 1, p] > 0.02 * W or Wpt[tectal - 1, p] > 0.02 * W):
                newweight[tectal, p] = 0.01 * W

    # CALCULATE WEIGHT CHANGE
    weightchange = newweight - Wpt
    return weightchange


def tabulate_weight_matrix():
    table = np.zeros([nR * nT, 4])
    row = 0
    deltaw = weight_change()
    for p in range(rmin, rmax + 1):
        for tectal in range(tmin, tmax + 1):
            table[row, 0] = p
            table[row, 1] = tectal
            table[row, 2] = Wpt[tectal, p]
            if deltaw[tectal, p] >= 0:
                table[row, 3] = 1
            else:
                table[row, 3] = 0
            row += 1
    return table


plotcount = 1
for iterations in range(Iterations+1):
    # PLOTS
    if iterations % 100 == 0 or iterations == 0:
        plt.subplot(3, 7, plotcount)
        plot = tabulate_weight_matrix()
        plt.scatter(plot[:, 1], plot[:, 0], s=(plot[:, 2]) * 20, marker='s', c=(plot[:, 3]), cmap='Greys',
                    edgecolors='k')
        plt.clim(0, 1)
        plt.ylabel('Presynaptic Cell Number')
        plt.xlabel('Postsynaptic Cell Number')
        plt.xlim([tmin - 1, tmax])
        plt.ylim([rmin - 1, rmax])
        plt.title('%d iterations' %(iterations))
        plotcount += 1


    deltaW = weight_change()
    Wpt += deltaW

    Qtm = np.dot(Wpt, normalisedCpm)
    for t in range(td):
        deltaconc = conc_change(Ctm, 'tectal')
        Ctm += (deltaconc * deltat)
    normalisedCtm = normalise(Ctm, 'tectal')



####################### END #########################

end = time.time()
elapsed = end - start
print('Time elapsed: ', elapsed, 'seconds')

params = {'font.size': '10'}
plt.rcParams.update(params)
plt.show()