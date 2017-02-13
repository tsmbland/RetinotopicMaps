import numpy as np
import random
import time
import sys

start = time.time()

#################### PARAMETERS #####################

# General
Iterations = 100  # number of weight iterations
NRdim1 = 20  # initial number of retinal cells
NRdim2 = 20
NTdim1 = 20  # initial number of tectal cells
NTdim2 = 20
Mdim1 = 3  # number of markers
Mdim2 = 3

# Establishment of initial contacts
n0 = 10  # number of initial random contact
NLdim1 = 15  # sets initial bias
NLdim2 = 15

# Presynaptic concentrations
a = 0.006  # (or 0.003) #decay constant
d = 0.3  # diffusion length constant
E = 0.01  # concentration elimination threshold
Q = 100.  # release of markers from source
stab = 0.1  # retinal stability threshold

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

# Output
TRout = 5  # temporal resoultion of output files

################### VARIABLES ###################

Rmindim1 = 1
Rmaxdim1 = NRdim1
Rmindim2 = 1
Rmaxdim2 = NRdim2
Tmindim1 = 1
Tmaxdim1 = NTdim1
Tmindim2 = 1
Tmaxdim2 = NTdim2

M = Mdim1 * Mdim2

Wpt = np.zeros([Iterations + 1, NTdim1 + 2, NTdim2 + 2, NRdim1 + 2,
                NRdim2 + 2])  # synaptic strength between a presynaptic cell and a postsynaptic cell

Qpm = np.zeros([M, NRdim1 + 2, NRdim2 + 2])  # presence of marker sources along retina
Qtm = np.zeros([M, NTdim1 + 2, NTdim2 + 2])  # axonal flow of molecules into postsymaptic cells

Cpm = np.zeros([M, NRdim1 + 2, NRdim2 + 2])  # concentration of a molecule in a presynaptic cell
Ctm = np.zeros([M, Iterations + 1, NTdim1 + 2, NTdim2 + 2])  # concentration of a molecule in a postsynaptic cell
NormalisedCpm = np.zeros(
    [M, NRdim1 + 2, NRdim2 + 2])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
NormalisedCtm = np.zeros(
    [M, NTdim1 + 2, NTdim2 + 2])  # normalised (by marker conc.) marker concentration in a postsynaptic cell

NCp = np.zeros([NRdim1 + 2, NRdim2 + 2])
NCt = np.zeros([NTdim1 + 2, NTdim2 + 2])

Currentiteration = 0


################### FUNCTIONS #####################

def setmarkerlocations():
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


def updateNc():
    # Presynaptic neuron map
    nmp = np.zeros([NRdim1 + 2, NRdim2 + 2])
    nmp[Rmindim1:Rmaxdim1 + 1, Rmindim2:Rmaxdim2 + 1] = 1

    # Tectal neuron map
    nmt = np.zeros([NTdim1 + 2, NTdim2 + 2])
    nmt[Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1] = 1

    # Presynaptic neighbour count
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            NCp[rdim1, rdim2] = nmp[rdim1 + 1, rdim2] + nmp[rdim1 - 1, rdim2] + nmp[rdim1, rdim2 + 1] + nmp[
                rdim1, rdim2 - 1]

    # Tectal neighbour count
    for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
        for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
            NCt[tdim1, tdim2] = nmt[tdim1 + 1, tdim2] + nmt[tdim1 - 1, tdim2] + nmt[tdim1, tdim2 + 1] + nmt[
                tdim1, tdim2 - 1]


def setretinalconcs():
    averagemarkerchange = 1
    while averagemarkerchange > stab:
        concchange = np.zeros([M, len(Cpm[0, :, 0]), len(Cpm[0, 0, :])])
        for m in range(M):
            for dim1 in range(Rmindim1, Rmaxdim1 + 1):
                for dim2 in range(Rmindim2, Rmaxdim2 + 1):
                    concchange[m, dim1, dim2] = (-a * Cpm[m, dim1, dim2] + d * (
                        Cpm[m, dim1, dim2 + 1] + Cpm[m, dim1, dim2 - 1] + Cpm[m, dim1 + 1, dim2] +
                        Cpm[m, dim1 - 1, dim2] - NCp[dim1, dim2] * Cpm[m, dim1, dim2]) + Qpm[
                                                     m, dim1, dim2])

        averagemarkerchange = (sum(sum(sum(concchange))) / sum(sum(sum(Cpm)))) * 100
        Cpm[:, :, :] += (concchange * deltat)


def updatetectalconcs():
    if Currentiteration > 0:
        Ctm[:, Currentiteration, :, :] = Ctm[:, Currentiteration - 1, :, :]
    for t in range(td):
        for m in range(M):
            for dim1 in range(Rmindim1, Rmaxdim1 + 1):
                for dim2 in range(Rmindim2, Rmaxdim2 + 1):
                    Ctm[m, Currentiteration, dim1, dim2] += (-a * Ctm[m, Currentiteration, dim1, dim2] + d * (
                        Ctm[m, Currentiteration, dim1, dim2 + 1] + Ctm[m, Currentiteration, dim1, dim2 - 1] + Ctm[
                            m, Currentiteration, dim1 + 1, dim2] +
                        Ctm[m, Currentiteration, dim1 - 1, dim2] - NCt[dim1, dim2] * Ctm[
                            m, Currentiteration, dim1, dim2]) + Qtm[
                                                                 m, dim1, dim2]) * deltat


def normaliseCpm():
    markersum = np.sum(Cpm, axis=0)
    for m in range(M):
        for dim1 in range(Rmindim1, Rmaxdim1 + 1):
            for dim2 in range(Rmindim2, Rmaxdim2 + 1):
                NormalisedCpm[m, dim1, dim2] = Cpm[m, dim1, dim2] / markersum[dim1, dim2]
                if NormalisedCpm[m, dim1, dim2] < E:
                    NormalisedCpm[m, dim1, dim2] = 0


def normaliseCtm():
    markersum = np.sum(Ctm[:, Currentiteration, :, :], axis=0)
    for m in range(M):
        for dim1 in range(Tmindim1, Tmaxdim1 + 1):
            for dim2 in range(Tmindim2, Tmaxdim2 + 1):
                NormalisedCtm[m, dim1, dim2] = Ctm[m, Currentiteration, dim1, dim2] / markersum[dim1, dim2]
                if NormalisedCtm[m, dim1, dim2] < E:
                    NormalisedCtm[m, dim1, dim2] = 0


def initialconections(rdim1, rdim2):
    initialstrength = W / n0
    if int(rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) <= Tmaxdim1 - Tmindim1 + 1:
        if int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= Tmaxdim2 - Tmindim2 + 1:
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
    elif int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= Tmaxdim2 - Tmindim2 + 1:
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
    for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
        for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
            for m in range(M):
                Qtm[m, tdim1, tdim2] = sum(sum(NormalisedCpm[m, :, :] * Wpt[
                                                                        Currentiteration, tdim1, tdim2, :, :]))


def updateWeight():
    Spt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    deltaWpt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    totalSp = np.zeros([NRdim1 + 2, NRdim2 + 2])
    meanSp = np.zeros([NRdim1 + 2, NRdim2 + 2])
    deltaWsum = np.zeros([NRdim1 + 2, NRdim2 + 2])
    connections = np.zeros([NRdim1 + 2, NRdim2 + 2])

    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            synapses = np.array(
                np.nonzero(Wpt[Currentiteration - 1, Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1, rdim1, rdim2])) + 1
            connections[rdim1, rdim2] = len(synapses[0, :])
            for synapse in range(int(connections[rdim1, rdim2])):
                Spt[synapses[0, synapse], synapses[1, synapse], rdim1, rdim2] = sum(
                    np.minimum(NormalisedCpm[:, rdim1, rdim2],
                               NormalisedCtm[:, synapses[0, synapse], synapses[1, synapse]]))
            totalSp[rdim1, rdim2] = sum(sum(Spt[:, :, rdim1, rdim2]))

    # Calculate mean similarity
    meanSp[Rmindim1:Rmaxdim1 + 1, Rmindim2:Rmaxdim2 + 1] = (totalSp[Rmindim1:Rmaxdim1 + 1,
                                                            Rmindim2:Rmaxdim2 + 1] / connections[
                                                                                     Rmindim1:Rmaxdim1 + 1,
                                                                                     Rmindim2:Rmaxdim2 + 1]) - k

    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            # Calculate deltaW
            deltaWpt[Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1, rdim1, rdim2] = h * (
                Spt[Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1, rdim1, rdim2] - meanSp[rdim1, rdim2])

            synapses = np.array(
                np.nonzero(Wpt[Currentiteration - 1, Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1, rdim1, rdim2])) + 1
            for synapse in range(int(connections[rdim1, rdim2])):
                deltaWsum[rdim1, rdim2] += deltaWpt[synapses[0, synapse], synapses[1, synapse], rdim1, rdim2]

            # Update Weight
            Wpt[Currentiteration, :, :, rdim1, rdim2] = (Wpt[Currentiteration - 1, :, :, rdim1, rdim2] + deltaWpt[:, :,
                                                                                                         rdim1,
                                                                                                         rdim2]) / (
                                                            W + deltaWsum[rdim1, rdim2])


def removesynapses():
    Wpt[Currentiteration, Wpt[Currentiteration, :, :, :, :] < elim * W] = 0.


def addsynapses():
    synapses = np.array(
        np.nonzero(Wpt[Currentiteration, :, :, :, :]))

    for synapse in range(int(len(synapses[0, :]))):
        if Wpt[Currentiteration, synapses[0, synapse], synapses[1, synapse], synapses[2, synapse], synapses[
            3, synapse]] > sprout * W:

            if Wpt[Currentiteration, synapses[0, synapse] + 1, synapses[1, synapse], synapses[2, synapse], synapses[
                3, synapse]] == 0 and synapses[0, synapse] + 1 <= Tmaxdim1:
                Wpt[Currentiteration, synapses[0, synapse] + 1, synapses[1, synapse], synapses[2, synapse], synapses[
                    3, synapse]] = newW * W
            if Wpt[Currentiteration, synapses[0, synapse] - 1, synapses[1, synapse], synapses[2, synapse], synapses[
                3, synapse]] == 0 and synapses[0, synapse] - 1 >= Tmindim1:
                Wpt[Currentiteration, synapses[0, synapse] - 1, synapses[1, synapse], synapses[2, synapse], synapses[
                    3, synapse]] = newW * W
            if Wpt[Currentiteration, synapses[0, synapse], synapses[1, synapse] + 1, synapses[2, synapse], synapses[
                3, synapse]] == 0 and synapses[1, synapse] + 1 <= Tmaxdim2:
                Wpt[Currentiteration, synapses[0, synapse], synapses[1, synapse] + 1, synapses[2, synapse], synapses[
                    3, synapse]] = newW * W
            if Wpt[Currentiteration, synapses[0, synapse], synapses[1, synapse] - 1, synapses[2, synapse], synapses[
                3, synapse]] == 0 and synapses[1, synapse] - 1 >= Tmindim2:
                Wpt[Currentiteration, synapses[0, synapse], synapses[1, synapse] - 1, synapses[2, synapse], synapses[
                    3, synapse]] = newW * W


######################## ALGORITHM ##########################

# MARKER LOCATIONS

setmarkerlocations()

# PRESYNAPTIC CONCENTRATIONS

updateNc()
setretinalconcs()
normaliseCpm()

# INITIAL CONNECTIONS

for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
    for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
        initialconections(rdim1, rdim2)

# INITIAL CONCENTRATIONS

updateQtm()
updatetectalconcs()
normaliseCtm()

# ITERATIONS

for iteration in range(1, Iterations + 1):
    Currentiteration += 1
    updateWeight()
    removesynapses()
    addsynapses()

    updateQtm()
    updatetectalconcs()
    normaliseCtm()

    sys.stdout.write('\r%i percent' % (iteration * 100 / Iterations))
    sys.stdout.flush()

#################### EXPORT DATA #################

np.save('../Temporary Data/Weightmatrix', Wpt[0:Iterations + 2:TRout, :, :, :, :])
np.save('../Temporary Data/Retinal Concentrations', Cpm)
np.save('../Temporary Data/Tectal Concentrations', Ctm[:, 0:Iterations + 2:TRout, :, :])

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
