import numpy as np
import random
import time
import sys

start = time.time()

#################### PARAMETERS #####################

# General
NRdim1 = 20  # initial number of retinal cells
NRdim2 = 20
NTdim1 = 20  # initial number of tectal cells
NTdim2 = 20
Mdim1 = 3  # number of markers
Mdim2 = 3

# Presynaptic concentrations
a = 0.006  # (or 0.003) #decay constant
d = 0.3  # diffusion length constant
E = 0.01  # concentration elimination threshold
Q = 100.  # release of markers from source
stab = 0.1  # retinal stability threshold

# Establishment of initial contacts
n0 = 10  # number of initial random contact
NLdim1 = 15  # sets initial bias
NLdim2 = 15

# Tectal concentrations
deltat = 0.5  # time step
td = 10  # number of concentration iterations per weight iteration

# Synaptic modification
W = 1.  # total strength available to each presynaptic fibre
h = 0.01  # ???
k = 0.03  # ???
elim = 0.005  # elimination threshold
newW = 0.01  # weight of new synapses
sprout = 0.02  # sprouting threshold

Iterations = 500  # number of weight iterations

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

Wpt = np.zeros([Iterations + 1, NTdim1 + 2, NTdim2 + 2, NRdim1 + 2,
                NRdim2 + 2])  # synaptic strength between a presynaptic cell and a postsynaptic cell

Qpm = np.zeros([M, NRdim1 + 2, NRdim2 + 2])  # presence of marker sources along retina
Qtm = np.zeros([M, NTdim1 + 2, NTdim2 + 2])  # axonal flow of molecules into postsymaptic cells

Cpm = np.zeros([M, NRdim1 + 2, NRdim2 + 2])  # concentration of a molecule in a presynaptic cell
Ctm = np.zeros([M, NTdim1 + 2, NTdim2 + 2])  # concentration of a molecule in a postsynaptic cell
normalisedCpm = np.zeros(
    [M, NRdim1 + 2, NRdim2 + 2])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
normalisedCtm = np.zeros(
    [M, NTdim1 + 2, NTdim2 + 2])  # normalised (by marker conc.) marker concentration in a postsynaptic cell

NCp = np.zeros([NRdim1 + 2, NRdim2 + 2])
NCt = np.zeros([NTdim1 + 2, NTdim2 + 2])

currentiteration = 0


################### FUNCTIONS #####################

def updateNc():
    # Presynaptic neuron map
    nmp = np.zeros([NRdim1 + 2, NRdim2 + 2])
    nmp[rmindim1:rmaxdim1 + 1, rmindim2:rmaxdim2 + 1] = 1

    # Tectal neuron map
    nmt = np.zeros([NTdim1 + 2, NTdim2 + 2])
    nmt[tmindim1:tmaxdim1 + 1, tmindim2:tmaxdim2 + 1] = 1

    # Presynaptic neighbour count
    for rdim1 in range(rmindim1, rmaxdim1 + 1):
        for rdim2 in range(rmindim2, rmaxdim2 + 1):
            NCp[rdim1, rdim2] = nmp[rdim1 + 1, rdim2] + nmp[rdim1 - 1, rdim2] + nmp[rdim1, rdim2 + 1] + nmp[
                rdim1, rdim2 - 1]

    # Tectal neighbour count
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            NCt[tdim1, tdim2] = nmt[tdim1 + 1, tdim2] + nmt[tdim1 - 1, tdim2] + nmt[tdim1, tdim2 + 1] + nmt[
                tdim1, tdim2 - 1]


def conc_change(concmatrix, layer):
    # Layer
    if layer == 'presynaptic':
        dim1start = rmindim1
        dim1end = rmaxdim1
        dim2start = rmindim2
        dim2end = rmaxdim2
        Qmatrix = Qpm
        nc = NCp
    elif layer == 'tectal':
        dim1start = tmindim1
        dim1end = tmaxdim1
        dim2start = tmindim2
        dim2end = tmaxdim2
        Qmatrix = Qtm
        nc = NCt

    # Conc change
    concchange = np.zeros([M, len(concmatrix[0, :, 0]), len(concmatrix[0, 0, :])])
    for m in range(M):
        for dim1 in range(dim1start, dim1end + 1):
            for dim2 in range(dim2start, dim2end + 1):
                concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (
                    concmatrix[m, dim1, dim2 + 1] + concmatrix[m, dim1, dim2 - 1] + concmatrix[m, dim1 + 1, dim2] +
                    concmatrix[m, dim1 - 1, dim2] - nc[dim1, dim2] * concmatrix[m, dim1, dim2]) + Qmatrix[
                                                 m, dim1, dim2])

    return concchange


def normalise(concmatrix, layer):
    # Layer
    if layer == 'presynaptic':
        dim1start = rmindim1
        dim1end = rmaxdim1
        dim2start = rmindim1
        dim2end = rmaxdim2
    elif layer == 'tectal':
        dim1start = tmindim1
        dim1end = tmaxdim1
        dim2start = tmindim1
        dim2end = tmaxdim2

    # Matrix size
    lengthdim1 = len(concmatrix[0, :, 0])
    lengthdim2 = len(concmatrix[0, 0, :])

    # Marker sum
    markersum = np.zeros([lengthdim1, lengthdim2])
    for dim1 in range(dim1start, dim1end + 1):
        for dim2 in range(dim2start, dim2end + 1):
            markersum[dim1, dim2] = sum(concmatrix[:, dim1, dim2])

    # Normalisation
    normalised = np.zeros([M, lengthdim1, lengthdim2])

    for m in range(M):
        for dim1 in range(dim1start, dim1end + 1):
            for dim2 in range(dim2start, dim2end + 1):
                normalised[m, dim1, dim2] = concmatrix[m, dim1, dim2] / markersum[dim1, dim2]
                if normalised[m, dim1, dim2] < E:
                    normalised[m, dim1, dim2] = 0

    return normalised


def initialconections(rdim1, rdim2):
    initialstrength = W / n0
    if int(rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) <= nTdim1:
        if int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= nTdim2:
            # Fits in both dimensions
            arrangement = np.zeros([NLdim1 * NLdim2])
            arrangement[0:n0] = initialstrength
            random.shuffle(arrangement)
            arrangement = np.reshape(arrangement, (NLdim1, NLdim2))
            Wpt[0, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: int(
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
            Wpt[0, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: int(
                rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) + 1,
            int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: NTdim2 + 1, rdim1,
            rdim2] = arrangement
    elif int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= nTdim2:
        # Doesn't fit into dim1 but fits into dim2
        arrangement = np.zeros([(NTdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1))) * NLdim2])
        arrangement[0:n0] = initialstrength
        random.shuffle(arrangement)
        arrangement = np.reshape(arrangement, (NTdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)), NLdim2))
        Wpt[0, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: NTdim1 + 1,
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
        Wpt[0, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: NTdim1 + 1,
        int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: NTdim2 + 1,
        rdim1,
        rdim2] = arrangement


def updateQtm():
    Qtm[:, :, :] = 0.
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            for m in range(M):
                Qtm[m, tdim1, tdim2] = sum(sum(normalisedCpm[m, :, :] * Wpt[
                                                                        currentiteration, tdim1, tdim2, :, :]))


def updateWeight():
    # SYNAPTIC WEIGHT

    Spt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    deltaWpt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    totalSp = np.zeros([NRdim1 + 2, NRdim2 + 2])
    meanSp = np.zeros([NRdim1 + 2, NRdim2 + 2])
    deltaWsum = np.zeros([NRdim1 + 2, NRdim2 + 2])
    connections = np.zeros([NRdim1 + 2, NRdim2 + 2])

    for rdim1 in range(rmindim1, rmaxdim1 + 1):
        for rdim2 in range(rmindim2, rmaxdim2 + 1):
            for tdim1 in range(tmindim1, tmaxdim1 + 1):
                for tdim2 in range(tmindim2, tmaxdim2 + 1):

                    # Calculate similarity
                    for m in range(M):
                        Spt[tdim1, tdim2, rdim1, rdim2] += min(normalisedCpm[m, rdim1, rdim2],
                                                               normalisedCtm[m, tdim1, tdim2])

                    # Count connections
                    if Wpt[currentiteration - 1, tdim1, tdim2, rdim1, rdim2] > 0.:
                        totalSp[rdim1, rdim2] += Spt[tdim1, tdim2, rdim1, rdim2]
                        connections[rdim1, rdim2] += 1

    # Calculate mean similarity
    meanSp[rmindim1:rmaxdim1 + 1, rmindim2:rmaxdim2 + 1] = (totalSp[rmindim1:rmaxdim1 + 1,
                                                            rmindim2:rmaxdim2 + 1] / connections[
                                                                                     rmindim1:rmaxdim1 + 1,
                                                                                     rmindim2:rmaxdim2 + 1]) - k

    for rdim1 in range(rmindim1, rmaxdim1 + 1):
        for rdim2 in range(rmindim2, rmaxdim2 + 1):
            # Calculate deltaW
            deltaWpt[tmindim1:tmaxdim1 + 1, tmindim2:tmaxdim2 + 1, rdim1, rdim2] = h * (
                Spt[tmindim1:tmaxdim1 + 1, tmindim2:tmaxdim2 + 1, rdim1, rdim2] - meanSp[rdim1, rdim2])

            for tdim1 in range(tmindim1, tmaxdim1 + 1):
                for tdim2 in range(tmindim2, tmaxdim2 + 1):

                    # Calculate deltaWsum
                    if Wpt[currentiteration - 1, tdim1, tdim2, rdim1, rdim2] > 0.:
                        deltaWsum[rdim1, rdim2] += deltaWpt[tdim1, tdim2, rdim1, rdim2]

            # Update Weight
            Wpt[currentiteration, :, :, rdim1, rdim2] = (Wpt[currentiteration - 1, :, :, rdim1, rdim2] + deltaWpt[:, :,
                                                                                                         rdim1,
                                                                                                         rdim2]) / (
                                                            W + deltaWsum[rdim1, rdim2])


def removesynapses():
    Wpt[currentiteration, Wpt[currentiteration, :, :, :, :] < elim * W] = 0.


def addsynapses():
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            for rdim1 in range(rmindim1, rmaxdim1 + 1):
                for rdim2 in range(rmindim2, rmaxdim2 + 1):
                    if Wpt[currentiteration, tdim1, tdim2, rdim1, rdim2] == 0 and (
                                            Wpt[currentiteration, tdim1 + 1, tdim2, rdim1, rdim2] > 0.02 * W or Wpt[
                                        currentiteration,
                                        tdim1 - 1, tdim2, rdim1, rdim2] > 0.02 * W or Wpt[currentiteration,
                                                                                          tdim1, tdim2 + 1, rdim1, rdim2] > 0.02 * W or
                                    Wpt[currentiteration,
                                        tdim1, tdim2 - 1, rdim1, rdim2] > 0.02 * W):
                        Wpt[currentiteration, tdim1, tdim2, rdim1, rdim2] = newW * W


######################## ALGORITHM ##########################

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

updateNc()
averagemarkerchange = 1
while averagemarkerchange > stab:
    deltaconc = conc_change(Cpm, 'presynaptic')
    averagemarkerchange = (sum(sum(sum(deltaconc))) / sum(sum(sum(Cpm)))) * 100
    Cpm += (deltaconc * deltat)
normalisedCpm = normalise(Cpm, 'presynaptic')

# INITIAL CONNECTIONS

for rdim1 in range(rmindim1, rmaxdim1 + 1):
    for rdim2 in range(rmindim2, rmaxdim2 + 1):
        initialconections(rdim1, rdim2)

# INITIAL CONCENTRATIONS

updateQtm()
for t in range(td):
    deltaconc = conc_change(Ctm, 'tectal')
    Ctm += (deltaconc * deltat)
normalisedCtm = normalise(Ctm, 'tectal')

# ITERATIONS

for iteration in range(1, Iterations + 1):
    currentiteration += 1
    updateWeight()
    removesynapses()
    addsynapses()

    updateQtm()
    for t in range(td):
        deltaconc = conc_change(Ctm, 'tectal')
        Ctm += (deltaconc * deltat)
    normalisedCtm = normalise(Ctm, 'tectal')

    sys.stdout.write('\r%i percent' % (iteration * 100 / Iterations))
    sys.stdout.flush()

#################### EXPORT DATA #################

np.save('Weightmatrix', Wpt)

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
