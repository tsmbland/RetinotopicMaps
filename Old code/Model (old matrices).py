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
stab = 0.1 # stability threshold

# Establishment of initial contacts
n0 = 8  # number of initial random contact
NL = 60  # sets initial bias

# Tectal concentrations
deltat = 0.1  # time step
td = 5  # number of iterations???

# Synaptic modification
W = 1  # total strength available to each presynaptic fibre
h = 0.01  # ???
k = 0.03  # ???
elim = 0.005 #elimination threshold





################### VARIABLES ###################

Wpt = np.zeros([NR, NT])  # synaptic strength between a presynaptic cell and a postsynaptic cell
deltaWpt = np.zeros([NR,NT]) #change in synaptic weight between a presynaptic cell and a postsynaptic cell
deltaWsum = np.zeros([NR])

Qpm = np.zeros([NR, M])  # presence of marker sources along retina
Qtm = np.zeros([NT, M])  # axonal flow of molecules into postsymaptic cells

Cpm = np.zeros([NR, M])  # concentration of a molecule in a presynaptic cell
Ctm = np.zeros([NT,M]) #concentration of a molecule in a postsynaptic cell
# normalisedCpm = np.zeros([NR,M]) #normalised (by marker conc.) marker concentration  in a presynaptic cell
# normalisedCtm = np.zeros([NT,M]) #normalised (by marker conc.) marker concentration in a postsynaptic cell

Spt = np.zeros([NR,NT]) #similarity between a presynaptic and a postsynaptic cell
totalSp = np.zeros([NR])
connections = np.zeros([NR]) #number of connections for a presynaptic cell
meanSp = np.zeros([NR]) #mean similarity for a presynaptic cell







################## PRESYNAPTIC #####################


# MARKER LOCATIONS
markerspacing = NR / (M - 1)
location = 0
for m in range(M - 1):
    Qpm[location, m] = Q
    location = location + markerspacing
Qpm[NR - 1, M - 1] = Q



# PRESYNAPTIC CONCENTRATIONS
def conc_change(concmatrix, layer):
    if layer == 'presynaptic':
        Qmatrix = Qpm
    elif layer == 'tectal':
        Qmatrix = Qtm

    length = len(concmatrix[:, 0])
    conc_change = np.zeros([length, M])
    for m in range(M):
        conc_change[0, m] = (-a * concmatrix[0, m] + d * (-concmatrix[0, m] + concmatrix[1, m]) + Qmatrix[0, m])
        conc_change[length - 1, m] = (
        -a * concmatrix[length - 1, m] + d * (concmatrix[length - 2, m] - concmatrix[length - 1, m]) + Qmatrix[
            length - 1, m])
        for cell in range(1, length - 1):
            conc_change[cell, m] = (-a * concmatrix[cell, m] + d * (
            concmatrix[cell - 1, m] - 2 * concmatrix[cell, m] + concmatrix[cell + 1, m]) + Qmatrix[cell, m])
    return conc_change


averagemarkerchange = 1
while averagemarkerchange > stab:
    deltaconc = conc_change(Cpm, 'presynaptic')
    averagemarkerchange = sum(sum(deltaconc))/sum(sum(Cpm))*100
    Cpm = Cpm + deltaconc*deltat



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
            concmatrix[p, m] = normalised[p, m]



normalise(Cpm)
# doesn't quite work, normalise function must be wrong?






#################### POSTSYNAPTIC ######################


# INITIAL CONNECTIONS
initialstrength = W / n0
arrangement = np.zeros([NL])
arrangement[0:n0] = initialstrength

for p in range(NR):
    random.shuffle(arrangement)
    Wpt[p, np.ceil(p * ((NT - NL) / NR)): np.ceil(p * ((NT - NL) / NR) + NL)] = arrangement

# would be good to adapt this so that initial connections are normally distributed



# AXONAL FLOW
def setQtm():
    for m in range(M):
        for tectalcell in range(NT):
            axonalflow = 0
            for presynapticcell in range(NR):
                axonalflow = axonalflow + Cpm[presynapticcell, m] * Wpt[presynapticcell, tectalcell]

            Qtm[tectalcell, m] = axonalflow

setQtm()

# ITERATIONS

for t in range(100):
    setQtm()

    # POSTSYNAPTIC CONCENTRATIONS
    for t in range(td):
        deltaconc = conc_change(Ctm, 'tectal')
        #averagemarkerchange = sum(sum(deltaconc)) / sum(sum(Ctm)) * 100
        Ctm = Ctm + deltaconc*deltat


    # NORMALISED POSTSYNAPTIC CONCENTRATIONS
    normalise(Ctm)


    # SYNAPTIC WEIGHT
    for p in range(NR):
        for tectal in range(NT):
            # calculate similarity
            Spt[p, tectal] = scipy.spatial.distance.minkowski(Cpm[p, :], Ctm[tectal, :] ,2)
            if Wpt[p, tectal] != 0:
                totalSp[p] = totalSp[p] + Spt[p, tectal]
                connections[p] = connections[p] + 1
        # calculate mean similarity
        meanSp[p] = totalSp[p] / connections[p] - k

        for tectal in range(NT):
            # calculate deltaW
            deltaWpt[p,tectal] = h * (Spt[p, tectal] - meanSp[p])

        # calculate deltaWsum
        deltaWsum[p] = sum(deltaWpt[p, :])

        for tectal in range(NT):
            # calculate new W
            Wpt[p, tectal] = (Wpt[p, tectal] + deltaWpt[p, tectal]) * W / (W + deltaWsum[p])

    # REMOVE SYNAPSES
    for p in range(NR):
        for tectal in range(NT):
            if Wpt[p, tectal] < elim*W:
                Wpt[p, tectal] = 0

    # ADD NEW SYNAPSES
    for p in range(NR):
        if Wpt[p,0] == 0 and Wpt[p,1] > 0.02*W:
            Wpt[p, 0] = 0.01 * W
        if Wpt[p,NT-1] == 0 and Wpt[p,NT-2] > 0.02*W:
            Wpt[p, NT-1] = 0.01 * W
        for tectal in range(1, NT-1):
            if Wpt[p, tectal] == 0 and (Wpt[p, tectal + 1] > 0.02*W or Wpt[p, tectal - 1] > 0.02*W):
                Wpt[p, tectal] = 0.01*W

####################### OUTPUT #######################

#TIME ELAPSED
end = time.time()
elapsed = end - start
print('Time elapsed: ', elapsed, 'seconds')


# PLOT NORMALISED PRESYNAPTIC CONCENTRATIONS
#for m in range(M):
    #plt.plot(range(1, NR + 1), Cpm[:, m])
#plt.xlabel('Presynaptic Cell Number')
#plt.ylabel('Normalised Marker Concentration')
#plt.show()


# PLOT NORMALISED TECTAL CONCENTRATIONS
for m in range(M):
    plt.plot(range(1, NR + 1), Ctm[:, m])
plt.xlabel('Tectal Cell Number')
plt.ylabel('Normalised Marker Concentration')
plt.show()


# PLOT SYNAPTIC WEIGHT FIGURE