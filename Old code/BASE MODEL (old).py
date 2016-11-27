import numpy as np
import matplotlib.pyplot as plt
import random
import time
import scipy.spatial

start = time.time()




#################### PARAMETERS #####################

# General
NR = 80  # number of retinal cells
NT = 80  # number of tectal cells
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

Wpt = np.zeros([NT, NR])  # synaptic strength between a presynaptic cell and a postsynaptic cell
deltaWpt = np.zeros([NT, NR])  # change in synaptic weight between a presynaptic cell and a postsynaptic cell
deltaWsum = np.zeros([NR])

Qpm = np.zeros([NR, M])  # presence of marker sources along retina
Qtm = np.zeros([NT, M])  # axonal flow of molecules into postsymaptic cells

Cpm = np.zeros([NR, M])  # concentration of a molecule in a presynaptic cell
Ctm = np.zeros([NT, M])  # concentration of a molecule in a postsynaptic cell
normalisedCpm = np.zeros([NR, M])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
normalisedCtm = np.zeros([NT, M])  # normalised (by marker conc.) marker concentration in a postsynaptic cell

Spt = np.zeros([NT, NR])  # similarity between a presynaptic and a postsynaptic cell
meanSp = np.zeros([NR])  # mean similarity for a presynaptic cell




################## RETINA #####################


# MARKER LOCATIONS
markerspacing = NR / (M - 1)
location = 0
for m in range(M - 1):
    Qpm[location, m] = Q
    location += markerspacing
Qpm[NR - 1, M - 1] = Q


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





#################### TECTUM ######################


# INITIAL CONNECTIONS
initialstrength = W / n0
arrangement = np.zeros([NL])
arrangement[0:n0] = initialstrength

for p in range(NR):
    random.shuffle(arrangement)
    Wpt[np.ceil(p * ((NT - NL) / NR)): np.ceil(p * ((NT - NL) / NR) + NL), p] = arrangement

# would be good to adapt this so that initial connections are normally distributed



# ITERATIONS

averagemarkerchange = 1
iterations = 0
while iterations < 100:

    # AXONAL FLOW
    Qtm = np.dot(Wpt, normalisedCpm)

    # POSTSYNAPTIC CONCENTRATIONS
    for t in range(td):
        deltaconc = conc_change(Ctm, 'tectal')
        averagemarkerchange = sum(sum(deltaconc)) / sum(sum(Ctm)) * 100
        Ctm += (deltaconc * deltat)

    # NORMALISED POSTSYNAPTIC CONCENTRATIONS
    normalisedCtm = normalise(Ctm)


    # SYNAPTIC WEIGHT
    for p in range(NR):
        totalSp = 0
        connections = 0
        deltaWsum = np.zeros([NR])
        deltaWpt = np.zeros([NT, NR])

        for tectal in range(NT):
            # calculate similarity
            Spt[tectal, p] = 1 - scipy.spatial.distance.cosine(normalisedCpm[p, :], normalisedCtm[tectal, :])
            if Wpt[tectal, p] > 0:
                totalSp += Spt[tectal, p]
                connections += 1
        # calculate mean similarity
        meanSp[p] = (totalSp / connections) - k

        for tectal in range(NT):
            # calculate deltaW
            deltaWpt[tectal, p] = h * (Spt[tectal, p] - meanSp[p])
            # calculate deltaWsum
            if Wpt[tectal, p] > 0:
                deltaWsum[p] += deltaWpt[tectal,p]


        for tectal in range(NT):
            # calculate new W
            Wpt[tectal, p] = (Wpt[tectal, p] + deltaWpt[tectal, p]) * W / (W + deltaWsum[p])
            # REMOVE SYNAPSES
            if Wpt[tectal, p] < elim * W:
                Wpt[tectal, p] = 0

        # ADD NEW SYNAPSES
        if Wpt[0, p] == 0 and Wpt[1, p] > 0.02 * W:
            Wpt[0, p] = 0.01 * W
        if Wpt[NT - 1, p] == 0 and Wpt[NT - 2, p] > 0.02 * W:
            Wpt[NT - 1, p] = 0.01 * W
        for tectal in range(1, NT - 1):
            if Wpt[tectal, p] == 0 and (Wpt[tectal + 1, p] > 0.02 * W or Wpt[tectal - 1, p] > 0.02 * W):
                Wpt[tectal, p] = 0.01 * W

    iterations += 1



####################### OUTPUT #######################
print(iterations, 'iterations')

# TIME ELAPSED
end = time.time()
elapsed = end - start
print('Time elapsed: ', elapsed, 'seconds')

# PLOT PRESYNAPTIC CONCENTRATIONS
for m in range(M):
    plt.plot(range(1, NR + 1), Cpm[:, m])
plt.xlabel('Presynaptic Cell Number')
plt.ylabel('Marker Concentration')
plt.show()

# PLOT TECTAL CONCENTRATIONS
for m in range(M):
    plt.plot(range(1, NT + 1), Ctm[:, m])
plt.xlabel('Postsynaptic Cell Number')
plt.ylabel('Marker Concentration')
plt.show()


# PLOT SYNAPTIC WEIGHT FIGURE
plt.pcolor(Wpt)
plt.xlabel('Presynaptic Cell Number')
plt.ylabel('Postsynaptic Cell Number')
plt.show()
