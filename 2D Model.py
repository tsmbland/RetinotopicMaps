import numpy as np
import matplotlib.pyplot as plt
import random
import time

start = time.time()

#################### PARAMETERS #####################

# General
NRdim1 = 10  # initial number of retinal cells
NRdim2 = 10
NTdim1 = 10  # initial number of tectal cells
NTdim2 = 10
M = 5  # number of markers

# Presynaptic concentrations
a = 0.006  # (or 0.003) #decay constant
d = 0.3  # diffusion length constant
E = 0.01  # synaptic elimination threshold
Q = 100  # release of markers from source
stab = 0.1  # retinal stability threshold

# Establishment of initial contacts
n0 = 16  # number of initial random contact
NL = 4  # sets initial bias

# Tectal concentrations
deltat = 0.1  # time step
td = 50  # number of concentration iterations per weight iteration

# Synaptic modification
W = 1  # total strength available to each presynaptic fibre
h = 0.01  # ???
k = 0.03  # ???
elim = 0.005  # elimination threshold
Iterations = 100  # number of weight iterations

################### VARIABLES ###################
nRdim1 = NRdim1  # present number of retinal cells (pre-surgery)
nRdim2 = NRdim2
nTdim1 = NTdim1  # present number of tectal cells (pre-surgery)
nTdim2 = NTdim2

Wpt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2,
                NRdim2 + 2])  # synaptic strength between a presynaptic cell and a postsynaptic cell
Qpm = np.zeros([M, NRdim1 + 2, NRdim2 + 2])  # presence of marker sources along retina
Qtm = np.zeros([M, NTdim1 + 2, NTdim2 + 2])  # axonal flow of molecules into postsymaptic cells
Cpm = np.zeros([M, NRdim1 + 2, NRdim2 + 2])  # concentration of a molecule in a presynaptic cell
Ctm = np.zeros([M, NTdim1 + 2, NTdim2 + 2])  # concentration of a molecule in a postsynaptic cell
normalisedCpm = np.zeros(
    [M, NRdim1 + 2, NRdim2 + 2])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
normalisedCtm = np.zeros(
    [M, NTdim1 + 2, NTdim2 + 2])  # normalised (by marker conc.) marker concentration in a postsynaptic cell

################## RETINA #####################

# MARKER LOCATIONS

Qpm[0, 1, 1] = Q
Qpm[1, 1, NRdim2] = Q
Qpm[2, NRdim1, 1] = Q
Qpm[3, NRdim1, NRdim2] = Q
Qpm[4, NRdim1 / 2, NRdim2 / 2] = Q


# PRESYNAPTIC CONCENTRATIONS

def conc_change(concmatrix, layer):
    # Matrix size
    ndim1 = len(concmatrix[0, :, 0]) - 2
    ndim2 = len(concmatrix[0, 0, :]) - 2

    # Neuron map
    nm = np.zeros([ndim1 + 2, ndim2 + 2])
    nm[1:-1, 1:-1] = 1

    # Neighbour Count
    nc = np.zeros([ndim1 + 2, ndim2 + 2])
    for dim1 in range(1, ndim1 + 1):
        for dim2 in range(1, ndim2 + 1):
            nc[dim1, dim2] = nm[dim1 + 1, dim2] + nm[dim1 - 1, dim2] + nm[dim1, dim2 + 1] + nm[dim1, dim2 - 1]

    # Qmatrix
    if layer == 'presynaptic':
        Qmatrix = Qpm
    elif layer == 'tectal':
        Qmatrix = Qtm

    # Conc change
    concchange = np.zeros([M, ndim1 + 2, ndim2 + 2])
    for m in range(M):
        for dim1 in range(1, ndim1 + 1):
            for dim2 in range(1, ndim2 + 1):
                concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (
                    concmatrix[m, dim1, dim2 + 1] + concmatrix[m, dim1, dim2 - 1] + concmatrix[m, dim1 + 1, dim2] +
                    concmatrix[m, dim1 - 1, dim2] - nc[dim1, dim2] * concmatrix[m, dim1, dim2]) + Qmatrix[
                                                 m, dim1, dim2])

    return concchange


averagemarkerchange = 1
while averagemarkerchange > stab:
    deltaconc = conc_change(Cpm, 'presynaptic')
    averagemarkerchange = (sum(sum(sum(deltaconc))) / sum(sum(sum(Cpm)))) * 100
    Cpm += (deltaconc * deltat)


# NORMALISED PRESYNAPTIC CONCENTRATIONS

def normalise(concmatrix):
    # Matrix size
    ndim1 = len(concmatrix[0, :, 0]) - 2
    ndim2 = len(concmatrix[0, 0, :]) - 2

    # Marker sum
    markersum = np.zeros([ndim1 + 2, ndim2 + 2])
    for dim1 in range(1, ndim1 + 1):
        for dim2 in range(1, ndim2 + 1):
            markersum[dim1, dim2] = sum(concmatrix[:, dim1, dim2])

    # Normalisation
    normalised = np.zeros([M, ndim1 + 2, ndim2 + 2])

    for m in range(M):
        for dim1 in range(1, ndim1 + 1):
            for dim2 in range(1, ndim2 + 1):
                normalised[m, dim1, dim2] = concmatrix[m, dim1, dim2] / markersum[dim1, dim2]
                if normalised[m, dim1, dim2] < E:
                    normalised[m, dim1, dim2] = 0

    return normalised


normalisedCpm = normalise(Cpm)


#################### CONNECTIONS ######################

# INITIAL CONNECTIONS

def initialconections():
    initialstrength = W / n0

    for rdim1 in range(1, NRdim1 + 1):
        for rdim2 in range(1, NRdim2 + 1):
            if int(rdim1 * ((NTdim1 - NL) / NRdim1) + NL) <= nTdim1:
                if int(rdim2 * ((NTdim2 - NL) / NRdim2) + NL) <= nTdim2:
                    arrangement = np.zeros([NL * NL])
                    arrangement[0:n0] = initialstrength
                    random.shuffle(arrangement)
                    arrangement = np.reshape(arrangement, (NL, NL))
                    Wpt[int(rdim1 * ((NTdim1 - NL) / NRdim1)) + 1: int(rdim1 * ((NTdim1 - NL) / NRdim1) + NL) + 1,
                    int(rdim2 * ((NTdim2 - NL) / NRdim2)) + 1: int(rdim2 * ((NTdim2 - NL) / NRdim2) + NL) + 1, rdim1,
                    rdim2] = arrangement
                else:
                    arrangement = np.zeros([(NTdim2 - int(rdim2 * ((NTdim2 - NL) / NRdim2))) * NL])
                    arrangement[0:n0] = initialstrength
                    random.shuffle(arrangement)
                    arrangement = np.reshape(arrangement, (NL, NTdim2 - int(rdim2 * ((NTdim2 - NL) / NRdim2))))
                    Wpt[int(rdim1 * ((NTdim1 - NL) / NRdim1)) + 1: int(rdim1 * ((NTdim1 - NL) / NRdim1) + NL) + 1,
                    int(rdim2 * ((NTdim2 - NL) / NRdim2)) + 1: NTdim2 + 1, rdim1,
                    rdim2] = arrangement
            else:
                arrangement = np.zeros([(NTdim1 - int(rdim1 * ((NTdim1 - NL) / NRdim1))) * NL])
                arrangement[0:n0] = initialstrength
                random.shuffle(arrangement)
                arrangement = np.reshape(arrangement, (NTdim1 - int(rdim1 * ((NTdim1 - NL) / NRdim1)), NL))
                Wpt[int(rdim1 * ((NTdim1 - NL) / NRdim1)) + 1: NTdim1 + 1,
                int(rdim2 * ((NTdim2 - NL) / NRdim2)) + 1: int(rdim2 * ((NTdim2 - NL) / NRdim2) + NL) + 1, rdim1,
                rdim2] = arrangement


initialconections()


# INITIAL CONCENTRATIONS
def updateQtm():
    Qtm[:, :, :] = 0
    for m in range(M):
        for tdim1 in range(1, NTdim1 + 1):
            for tdim2 in range(1, NTdim2 + 1):
                for rdim1 in range(1, NRdim1 + 1):
                    for rdim2 in range(1, NRdim2 + 1):
                        Qtm[m, tdim1, tdim2] += normalisedCpm[m, rdim1, rdim2] * Wpt[tdim1, tdim2, rdim1, rdim2]


updateQtm()

for t in range(td):
    deltaconc = conc_change(Ctm, 'tectal')
    Ctm += (deltaconc * deltat)
normalisedCtm = normalise(Ctm)


# ITERATIONS

def weight_change():
    # SYNAPTIC WEIGHT

    newweight = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    for rdim1 in range(1, nRdim1 + 1):
        for rdim2 in range(1, nRdim2 + 1):

            totalSp = 0
            connections = 0
            deltaWsum = 0
            deltaWpt = np.zeros([NTdim1 + 2, NTdim2 + 2])
            Spt = np.zeros([NTdim1 + 2, NTdim2 + 2])

            for tdim1 in range(1, nTdim1 + 1):
                for tdim2 in range(1, nTdim2 + 1):

                    # Calculate similarity
                    for m in range(M):
                        Spt[tdim1, tdim2] += min(normalisedCpm[m, rdim1, rdim2],
                                                 normalisedCtm[m, tdim1, tdim2])

                    # Count connections
                    if Wpt[tdim1, tdim2, rdim1, rdim2] > 0:
                        totalSp += Spt[tdim1, tdim2]
                        connections += 1

            # Calculate mean similarity
            meanSp = (totalSp / connections) - k

            for tdim1 in range(1, nTdim1 + 1):
                for tdim2 in range(1, nTdim2 + 1):

                    # Calculate deltaW
                    deltaWpt[tdim1, tdim2] = h * (Spt[tdim1, tdim2] - meanSp)

                    # Calculate deltaWsum
                    if Wpt[tdim1, tdim2, rdim1, rdim2] > 0:
                        deltaWsum += deltaWpt[tdim1, tdim2]

            for tdim1 in range(1, nTdim1 + 1):
                for tdim2 in range(1, nTdim2 + 1):

                    # Calculate new W
                    newweight[tdim1, tdim2, rdim1, rdim2] = (Wpt[tdim1, tdim2, rdim1, rdim2] + deltaWpt[
                        tdim1, tdim2]) * W / (W + deltaWsum)

                    # REMOVE SYNAPSES
                    if newweight[tdim1, tdim2, rdim1, rdim2] < elim * W:
                        newweight[tdim1, tdim2, rdim1, rdim2] = 0

            # ADD NEW SYNAPSES
            for tdim1 in range(1, nTdim1 + 1):
                for tdim2 in range(1, nTdim2 + 1):
                    if newweight[tdim1, tdim2, rdim1, rdim2] == 0 and (
                                    newweight[tdim1 + 1, tdim2, rdim1, rdim2] > 0.02 * W or newweight[
                                    tdim1 - 1, tdim2, rdim1, rdim2] > 0.02 * W or newweight[
                            tdim1, tdim2 + 1, rdim1, rdim2] > 0.02 * W or newweight[
                        tdim1, tdim2 - 1, rdim1, rdim2] > 0.02 * W):
                        newweight[tdim1, tdim2, rdim1, rdim2] = 0.01 * W

    # CALCULATE WEIGHT CHANGE
    weightchange = newweight - Wpt
    return weightchange


def field_centre():
    fieldcentre = np.zeros([2, NTdim1 + 2, NTdim2 + 2])
    for tdim1 in range(1, nTdim1 + 1):
        for tdim2 in range(1, nTdim2 + 1):
            totaldim1 = 0
            totaldim2 = 0
            weightsumdim1 = 0
            weightsumdim2 = 0

            for rdim1 in range(1, nRdim1 + 1):
                for rdim2 in range(1, nRdim2 + 1):
                    totaldim1 += rdim1 * Wpt[tdim1, tdim2, rdim1, rdim2]
                    weightsumdim1 += Wpt[tdim1, tdim2, rdim1, rdim2]
                    totaldim2 += rdim2 * Wpt[tdim1, tdim2, rdim1, rdim2]
                    weightsumdim2 += Wpt[tdim1, tdim2, rdim1, rdim2]

            if totaldim1 != 0:
                fieldcentre[0, tdim1, tdim2] = totaldim1 / weightsumdim1
            if totaldim2 != 0:
                fieldcentre[1, tdim1, tdim2] = totaldim2 / weightsumdim2

    return fieldcentre

def field_separation():
    for tdim1 in range(1, nTdim1 + 1):
        for tdim2 in range(1, nTdim2 + 1):
            if fieldcentre[0, tdim1, tdim2] != 0 and fieldcentre[1, tdim1, tdim2] != 0:
                pass
                # then this tectal cell has a field centre
                # look for nearest cells that don't have filed centres of zero
                # calculate field centre distance in both dimensions
                # find euclidian distance



for iterations in range(Iterations):
    deltaW = weight_change()
    Wpt += deltaW

    updateQtm()
    for t in range(td):
        deltaconc = conc_change(Ctm, 'tectal')
        Ctm += (deltaconc * deltat)
    normalisedCtm = normalise(Ctm)

fieldcentre = field_centre()

##################### PLOT #######################
plt.subplot(2, 1, 1)
for m in range(M):
    plt.plot(range(1, nTdim1 + 1), Ctm[m, 1:-1, 1])
plt.ylabel('Marker Concentration')
plt.xticks([], [])

plt.subplot(2, 1, 2)
plt.scatter(fieldcentre[0, 1:nTdim1 + 1, 1:nTdim2 + 1], fieldcentre[1, 1:nTdim1 + 1, 1:nTdim2 + 1])

###################### END ########################
end = time.time()
elapsed = end - start
print('Time elapsed: ', elapsed, 'seconds')

params = {'font.size': '10'}
plt.rcParams.update(params)
plt.show()
