import numpy as np
import matplotlib.pyplot as plt
import random
import time
import scipy.spatial
from SurgeryFunctions import *

start = time.time()




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




################## RETINA #####################


# MARKER LOCATIONS
markerspacing = nR / (M - 1)
location = 0
for m in range(M - 1):
    Qpm[location, m] = Q
    location += markerspacing
Qpm[nR - 1, M - 1] = Q


# PRESYNAPTIC CONCENTRATIONS
averagemarkerchange = 1
while averagemarkerchange > stab:
    deltaconc = conc_change(Cpm, 'presynaptic')
    averagemarkerchange = sum(sum(deltaconc)) / sum(sum(Cpm)) * 100
    Cpm += (deltaconc * deltat)



# NORMALISED PRESYNAPTIC CONCENTRATIONS
normalisedCpm = normalise(Cpm)





#################### TECTUM ######################


# INITIAL CONNECTIONS
initialconnections()
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
    for p in range(nR):
        totalSp = 0
        connections = 0
        deltaWsum = np.zeros([nR])
        deltaWpt = np.zeros([nT, nR])

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

    iterations += 1




################# INITIAL CONNECTIONS ################





###################### SURGERY #######################
nR = 40




#################### RECONNECT #######################

Wpt = np.zeros([nT, nR])
initialconnections()



#iterations(500)






####################### OUTPUT #######################
print(iterations, 'iterations')

# TIME ELAPSED
end = time.time()
elapsed = end - start
print('Time elapsed: ', elapsed, 'seconds')

# PLOT PRESYNAPTIC CONCENTRATIONS
for m in range(M):
    plt.plot(range(1, nR + 1), Cpm[0:nR, m])
plt.xlabel('Presynaptic Cell Number')
plt.ylabel('Marker Concentration')
plt.show()

# PLOT TECTAL CONCENTRATIONS
for m in range(M):
    plt.plot(range(1, nT + 1), Ctm[0:nT, m])
plt.xlabel('Postsynaptic Cell Number')
plt.ylabel('Marker Concentration')
plt.show()


# PLOT SYNAPTIC WEIGHT FIGURE
plt.pcolor(Wpt)
plt.xlabel('Presynaptic Cell Number')
plt.ylabel('Postsynaptic Cell Number')
plt.show()
