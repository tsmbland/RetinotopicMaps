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
Mdim1 = 2  # number of markers
Mdim2 = 2

# Presynaptic concentrations
a = 0.006  # (or 0.003) #decay constant
d = 0.3  # diffusion length constant
E = 0.01  # synaptic elimination threshold
Q = 100  # release of markers from source
stab = 0.1  # retinal stability threshold

# Establishment of initial contacts
n0 = 7  # number of initial random contact
NLdim1 = 7  # sets initial bias
NLdim2 = 7

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
rmindim1 = 1
rmaxdim1 = NRdim1
rmindim2 = 1
rmaxdim2 = NRdim2
tmindim1 = 1
tmaxdim1 = NTdim1
tmindim2 = 1
tmaxdim2 = NTdim2

nRdim1 = rmaxdim1 - rmindim1 + 1  # present number of retinal cells (pre-surgery)
nRdim2 = rmaxdim2 - rmindim2 + 1
nTdim1 = tmaxdim1 - tmindim1 + 1  # present number of tectal cells (pre-surgery)
nTdim2 = tmaxdim2 - tmindim2 + 1
M = Mdim1 * Mdim2

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

if Mdim1 > 1:
    markerspacingdim1 = NRdim1 / (Mdim1 - 1)
else:
    markerspacingdim1 = 0
if Mdim2 > 1:
    markerspacingdim2 = NRdim2 / (Mdim2 - 1)
else:
    markerspacingdim2 = 0

m = 0
locationdim1 = 1
locationdim2 = 1
for mdim2 in range(Mdim2 - 1):
    for mdim1 in range(Mdim1 - 1):
        Qpm[m, locationdim1, locationdim2] = Q
        locationdim1 += markerspacingdim1
        m += 1
    Qpm[m, NRdim1, locationdim2] = Q
    locationdim1 = 1
    locationdim2 += markerspacingdim2
    m += 1

for mdim1 in range(Mdim1 - 1):
    Qpm[m, locationdim1, NRdim2] = Q
    locationdim1 += markerspacingdim1
    m += 1
Qpm[m, NRdim1, NRdim2] = Q


# PRESYNAPTIC CONCENTRATIONS

def conc_change(concmatrix, layer):
    # Layer
    if layer == 'presynaptic':
        dim1start = rmindim1
        dim1end = rmaxdim1
        dim2start = rmindim1
        dim2end = rmaxdim2
        Qmatrix = Qpm
    elif layer == 'tectal':
        dim1start = tmindim1
        dim1end = tmaxdim1
        dim2start = tmindim1
        dim2end = tmaxdim2
        Qmatrix = Qtm

    # Matrix size
    lengthdim1 = len(concmatrix[0, :, 0])
    lengthdim2 = len(concmatrix[0, 0, :])

    # Neuron map
    nm = np.zeros([lengthdim1, lengthdim2])
    nm[dim1start:dim1end+1, dim2start:dim2end+1] = 1

    # Neighbour Count
    nc = np.zeros([lengthdim1, lengthdim2])
    for dim1 in range(dim1start, dim1end+1):
        for dim2 in range(dim2start, dim2end+1):
            nc[dim1, dim2] = nm[dim1 + 1, dim2] + nm[dim1 - 1, dim2] + nm[dim1, dim2 + 1] + nm[dim1, dim2 - 1]

    # Conc change
    concchange = np.zeros([M, lengthdim1, lengthdim2])
    for m in range(M):
        for dim1 in range(dim1start, dim1end+1):
            for dim2 in range(dim2start, dim2end+1):
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

def normalise(concmatrix, layer):
    # Layer
    if layer == 'presynaptic':
        dim1start = rmindim1
        dim1end = rmaxdim1
        dim2start = rmindim1
        dim2end = rmaxdim2
        Qmatrix = Qpm
    elif layer == 'tectal':
        dim1start = tmindim1
        dim1end = tmaxdim1
        dim2start = tmindim1
        dim2end = tmaxdim2
        Qmatrix = Qtm

    # Matrix size
    lengthdim1 = len(concmatrix[0, :, 0])
    lengthdim2 = len(concmatrix[0, 0, :])

    # Marker sum
    markersum = np.zeros([lengthdim1, lengthdim2])
    for dim1 in range(dim1start, dim1end+1):
        for dim2 in range(dim2start, dim2end+1):
            markersum[dim1, dim2] = sum(concmatrix[:, dim1, dim2])

    # Normalisation
    normalised = np.zeros([M, lengthdim1, lengthdim2])

    for m in range(M):
        for dim1 in range(dim1start, dim1end+1):
            for dim2 in range(dim2start, dim2end+1):
                normalised[m, dim1, dim2] = concmatrix[m, dim1, dim2] / markersum[dim1, dim2]
                if normalised[m, dim1, dim2] < E:
                    normalised[m, dim1, dim2] = 0

    return normalised


normalisedCpm = normalise(Cpm, 'presynaptic')


#################### CONNECTIONS ######################

# INITIAL CONNECTIONS

def initialconections(rdim1, rdim2):
    initialstrength = W / n0
    if int(rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) <= nTdim1:
        if int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= nTdim2:
            # Fits in both dimensions
            arrangement = np.zeros([NLdim1 * NLdim2])
            arrangement[0:n0] = initialstrength
            random.shuffle(arrangement)
            arrangement = np.reshape(arrangement, (NLdim1, NLdim2))
            Wpt[int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: int(
                rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) + 1,
            int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: int(
                rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) + 1, rdim1,
            rdim2] = arrangement
        else:
            # Fits in dim1 but not dim2
            arrangement = np.zeros([(NTdim2 - int(rdim2 * ((NTdim2 - NLdim2) / NRdim2))) * NLdim1])
            arrangement[0:n0] = initialstrength
            random.shuffle(arrangement)
            arrangement = np.reshape(arrangement, (NLdim1, NTdim2 - int(rdim2 * ((NTdim2 - NLdim2) / NRdim2))))
            Wpt[int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: int(
                rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) + 1,
            int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: NTdim2 + 1, rdim1,
            rdim2] = arrangement
    elif int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= nTdim2:
        # Doesn't fit into dim1 but fits into dim2
        arrangement = np.zeros([(NTdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1))) * NLdim2])
        arrangement[0:n0] = initialstrength
        random.shuffle(arrangement)
        arrangement = np.reshape(arrangement, (NTdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)), NLdim2))
        Wpt[int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: NTdim1 + 1,
        int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) + 1,
        rdim1,
        rdim2] = arrangement
    else:
        # Doesn't fit into either dimension
        arrangement = np.zeros([(NTdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1))) * (
            NTdim2 - int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)))])
        arrangement[0:n0] = initialstrength
        random.shuffle(arrangement)
        arrangement = np.reshape(arrangement, (
            NTdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)),
            NTdim2 - int(rdim2 * ((NTdim2 - NLdim2) / NRdim2))))
        Wpt[int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: NTdim1 + 1,
        int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: NTdim2 + 1,
        rdim1,
        rdim2] = arrangement


for rdim1 in range(rmindim1, rmaxdim1+1):
    for rdim2 in range(rmindim2, rmaxdim2+1):
        initialconections(rdim1, rdim2)


# INITIAL CONCENTRATIONS
def updateQtm():
    Qtm[:, :, :] = 0
    for m in range(M):
        for tdim1 in range(tmindim1, tmaxdim1+1):
            for tdim2 in range(tmindim2, tmaxdim2+1):
                for rdim1 in range(rmindim1, rmaxdim1+1):
                    for rdim2 in range(rmindim2, rmaxdim2+1):
                        Qtm[m, tdim1, tdim2] += normalisedCpm[m, rdim1, rdim2] * Wpt[tdim1, tdim2, rdim1, rdim2]


updateQtm()

for t in range(td):
    deltaconc = conc_change(Ctm, 'tectal')
    Ctm += (deltaconc * deltat)
normalisedCtm = normalise(Ctm, 'tectal')


# ITERATIONS

def weight_change():
    # SYNAPTIC WEIGHT

    newweight = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    for rdim1 in range(rmindim1, rmaxdim1+1):
        for rdim2 in range(rmindim2, rmaxdim2+1):

            totalSp = 0
            connections = 0
            deltaWsum = 0
            deltaWpt = np.zeros([NTdim1 + 2, NTdim2 + 2])
            Spt = np.zeros([NTdim1 + 2, NTdim2 + 2])

            for tdim1 in range(tmindim1, tmaxdim1+1):
                for tdim2 in range(tmindim2, tmaxdim2+1):

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

            for tdim1 in range(tmindim1, tmaxdim1+1):
                for tdim2 in range(tmindim2, tmaxdim2+1):

                    # Calculate deltaW
                    deltaWpt[tdim1, tdim2] = h * (Spt[tdim1, tdim2] - meanSp)

                    # Calculate deltaWsum
                    if Wpt[tdim1, tdim2, rdim1, rdim2] > 0:
                        deltaWsum += deltaWpt[tdim1, tdim2]

            for tdim1 in range(tmindim1, tmaxdim1+1):
                for tdim2 in range(tmindim2, tmaxdim2+1):

                    # Calculate new W
                    newweight[tdim1, tdim2, rdim1, rdim2] = (Wpt[tdim1, tdim2, rdim1, rdim2] + deltaWpt[
                        tdim1, tdim2]) * W / (W + deltaWsum)

                    # REMOVE SYNAPSES
                    if newweight[tdim1, tdim2, rdim1, rdim2] < elim * W:
                        newweight[tdim1, tdim2, rdim1, rdim2] = 0

            # ADD NEW SYNAPSES
            for tdim1 in range(tmindim1, tmaxdim1+1):
                for tdim2 in range(tmindim2, tmaxdim2+1):
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
    for tdim1 in range(tmindim1, tmaxdim1+1):
        for tdim2 in range(tmindim2, tmaxdim2+1):
            totaldim1 = 0
            totaldim2 = 0
            weightsumdim1 = 0
            weightsumdim2 = 0

            for rdim1 in range(rmindim1, rmaxdim1+1):
                for rdim2 in range(rmindim2, rmaxdim2+1):
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
    for tdim1 in range(tmindim1, tmaxdim1+1):
        for tdim2 in range(tmindim2, tmaxdim2+1):
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
    normalisedCtm = normalise(Ctm, 'tectal')

fieldcentre = field_centre()

##################### PLOT #######################
plt.subplot(3, 1, 1)
for m in range(M):
    plt.plot(range(1, nTdim2 + 1), Ctm[m, 5, 1:-1])
plt.ylabel('Marker Concentration')
plt.xticks([], [])

plt.subplot(3, 1, 2)
plot = np.swapaxes(Wpt[5, 1:-1, 5, 1:-1], 0, 1)
plt.pcolormesh(plot, cmap='Greys')
plt.ylabel('Retinal Cell Number (dimension 1)')
plt.xlabel('Tectal Cell Number (dimension 1)')

plt.subplot(3, 1, 3)
plt.scatter(fieldcentre[0, 1:nTdim1 + 1, 1:nTdim2 + 1], fieldcentre[1, 1:nTdim1 + 1, 1:nTdim2 + 1])

###################### END ########################
end = time.time()
elapsed = end - start
print('Time elapsed: ', elapsed, 'seconds')

params = {'font.size': '10'}
plt.rcParams.update(params)
plt.tight_layout()
plt.show()
