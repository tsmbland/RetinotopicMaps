import numpy as np
import random

#################### PARAMETERS #####################

# General
Iterations = 500  # number of weight iterations
NRdim1 = 20  # initial number of retinal cells
NRdim2 = 20
NTdim1 = 20  # initial number of tectal cells
NTdim2 = 20
Mdim1 = 3  # number of markers
Mdim2 = 3

# Establishment of initial contacts
n0 = 7  # number of initial random contact
NLdim1 = 15  # sets initial bias
NLdim2 = 15

# Mismatch surgery
sRmindim1 = 1
sRmaxdim1 = NRdim1
sRmindim2 = 1
sRmaxdim2 = NRdim2
sTmindim1 = 1
sTmaxdim1 = NTdim1 // 2
sTmindim2 = 1
sTmaxdim2 = NTdim2

# Development
dRmindim1 = NRdim1 // 4
dRmaxdim1 = 3 * NRdim1 // 4
dRmindim2 = NRdim2 // 4
dRmaxdim2 = 3 * NRdim2 // 4
dTmindim1 = 1
dTmaxdim1 = NTdim1 // 2
dTmindim2 = 1
dTmaxdim2 = NTdim2 // 2
dstep = 30  # time between growth iterations
td = 300  # time taken for new fibres to gain full strength

# Presynaptic concentrations
a = 0.006  # (or 0.003) #decay constant
d = 0.3  # diffusion length constant
E = 0.01  # concentration elimination threshold
Q = 100.  # release of markers from source
stab = 0.1  # retinal stability threshold

# Tectal concentrations
deltat = 0.5  # time step
tc = 10  # number of concentration iterations per weight iteration

# Synaptic modification
Wmax = 1.  # total (final) strength available to each presynaptic fibre
h = 0.01  # ???
k = 0.03  # ???
elim = 0.005  # elimination threshold
newW = 0.01  # weight of new synapses
sprout = 0.02  # sprouting threshold

# Output
TRout = 5  # temporal resoultion of output files

################### VARIABLES ###################

M = Mdim1 * Mdim2

Wpt = np.zeros([Iterations + 1, NTdim1 + 2, NTdim2 + 2, NRdim1 + 2,
                NRdim2 + 2])  # synaptic strength between a presynaptic cell and a postsynaptic cell
Wtot = np.zeros([NRdim1 + 1, NRdim2 + 1])  # total strength available to a fibre

Qpm = np.zeros([M, NRdim1 + 2, NRdim2 + 2])  # presence of marker sources along retina
Qtm = np.zeros([M, NTdim1 + 2, NTdim2 + 2])  # axonal flow of molecules into postsymaptic cells

Cpm = np.zeros([M, NRdim1 + 2, NRdim2 + 2])  # concentration of a molecule in a presynaptic cell
Ctm = np.zeros([M, Iterations + 1, NTdim1 + 2, NTdim2 + 2])  # concentration of a molecule in a postsynaptic cell
NormalisedCpm = np.zeros(
    [M, NRdim1 + 2, NRdim2 + 2])  # normalised (by marker conc.) marker concentration  in a presynaptic cell
NormalisedCtm = np.zeros(
    [M, NTdim1 + 2, NTdim2 + 2])  # normalised (by marker conc.) marker concentration in a postsynaptic cell

NCp = np.zeros([NRdim1 + 2, NRdim2 + 2])  # neighbour count for presynaptic cells
NCt = np.zeros([NTdim1 + 2, NTdim2 + 2])  # neighbour count for tectal cells

xFieldcentres = np.zeros(
    [2, Iterations + 1, NTdim1 + 1, NTdim2 + 1])  # expected field centres for tectal cells given parameters

Currentiteration = 0


################### FUNCTIONS #####################

def typestandard():
    global Rmindim1, Rmaxdim1, Rmindim2, Rmaxdim2, Tmindim1, Tmaxdim1, Tmindim2, Tmaxdim2
    Rmindim1 = 1
    Rmaxdim1 = NRdim1
    Rmindim2 = 1
    Rmaxdim2 = NRdim2
    Tmindim1 = 1
    Tmaxdim1 = NTdim1
    Tmindim2 = 1
    Tmaxdim2 = NTdim2


def typemismatchsurgery():
    global Rmindim1, Rmaxdim1, Rmindim2, Rmaxdim2, Tmindim1, Tmaxdim1, Tmindim2, Tmaxdim2
    Rmindim1 = sRmindim1
    Rmaxdim1 = sRmaxdim1
    Rmindim2 = sRmindim2
    Rmaxdim2 = sRmaxdim2
    Tmindim1 = sTmindim1
    Tmaxdim1 = sTmaxdim1
    Tmindim2 = sTmindim2
    Tmaxdim2 = sTmaxdim2


def typedevelopment():
    global Rmindim1, Rmaxdim1, Rmindim2, Rmaxdim2, Tmindim1, Tmaxdim1, Tmindim2, Tmaxdim2
    Rmindim1 = dRmindim1
    Rmaxdim1 = dRmaxdim1
    Rmindim2 = dRmindim2
    Rmaxdim2 = dRmaxdim2
    Tmindim1 = dTmindim1
    Tmaxdim1 = dTmaxdim1
    Tmindim2 = dTmindim2
    Tmaxdim2 = dTmaxdim2


def setWtot():
    Wtot[1:NRdim1 + 1, 1:NRdim2 + 1] = Wmax / td
    Wtot[Rmindim1:Rmaxdim1 + 1, Rmindim2:Rmaxdim2 + 1] = Wmax


def updateWtot():
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            if Wtot[rdim1, rdim2] < Wmax:
                Wtot[rdim1, rdim2] += Wmax / td


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
            for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
                for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
                    concchange[m, rdim1, rdim2] = (-a * Cpm[m, rdim1, rdim2] + d * (
                        Cpm[m, rdim1, rdim2 + 1] + Cpm[m, rdim1, rdim2 - 1] + Cpm[m, rdim1 + 1, rdim2] +
                        Cpm[m, rdim1 - 1, rdim2] - NCp[rdim1, rdim2] * Cpm[m, rdim1, rdim2]) + Qpm[
                                                       m, rdim1, rdim2])

        averagemarkerchange = (sum(sum(sum(concchange))) / sum(sum(sum(Cpm)))) * 100
        Cpm[:, :, :] += (concchange * deltat)


def updateretinalconcs():
    for t in range(tc):
        for m in range(M):
            for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
                for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
                    Cpm[m, rdim1, rdim2] += (-a * Cpm[m, rdim1, rdim2] + d * (
                        Cpm[m, rdim1, rdim2 + 1] + Cpm[m, rdim1, rdim2 - 1] + Cpm[
                            m, rdim1 + 1, rdim2] +
                        Cpm[m, rdim1 - 1, rdim2] - NCp[rdim1, rdim2] * Cpm[
                            m, rdim1, rdim2]) + Qpm[m, rdim1, rdim2]) * deltat


def updatetectalconcs():
    if Currentiteration > 0:
        Ctm[:, Currentiteration, :, :] = Ctm[:, Currentiteration - 1, :, :]
    for t in range(tc):
        for m in range(M):
            for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
                for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                    Ctm[m, Currentiteration, tdim1, tdim2] += (-a * Ctm[m, Currentiteration, tdim1, tdim2] + d * (
                        Ctm[m, Currentiteration, tdim1, tdim2 + 1] + Ctm[m, Currentiteration, tdim1, tdim2 - 1] + Ctm[
                            m, Currentiteration, tdim1 + 1, tdim2] +
                        Ctm[m, Currentiteration, tdim1 - 1, tdim2] - NCt[tdim1, tdim2] * Ctm[
                            m, Currentiteration, tdim1, tdim2]) + Qtm[
                                                                   m, tdim1, tdim2]) * deltat


def normaliseCpm():
    NormalisedCpm[:, :, :] = 0
    markersum = np.sum(Cpm, axis=0)
    for m in range(M):
        for dim1 in range(Rmindim1, Rmaxdim1 + 1):
            for dim2 in range(Rmindim2, Rmaxdim2 + 1):
                NormalisedCpm[m, dim1, dim2] = Cpm[m, dim1, dim2] / markersum[dim1, dim2]
                if NormalisedCpm[m, dim1, dim2] < E:
                    NormalisedCpm[m, dim1, dim2] = 0
    NormalisedCpm[:, :, :] = np.nan_to_num(NormalisedCpm[:, :, :])


def normaliseCtm():
    NormalisedCtm[:, :, :] = 0
    markersum = np.sum(Ctm[:, Currentiteration, :, :], axis=0)
    for m in range(M):
        for dim1 in range(Tmindim1, Tmaxdim1 + 1):
            for dim2 in range(Tmindim2, Tmaxdim2 + 1):
                NormalisedCtm[m, dim1, dim2] = Ctm[m, Currentiteration, dim1, dim2] / markersum[dim1, dim2]
                if NormalisedCtm[m, dim1, dim2] < E:
                    NormalisedCtm[m, dim1, dim2] = 0
    NormalisedCtm[:, :, :] = np.nan_to_num(NormalisedCtm[:, :, :])


def connections(rdim1, rdim2):
    initialstrength = Wtot[rdim1, rdim2] / n0
    if int(rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) <= Tmaxdim1 - Tmindim1 + 1:
        if int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= Tmaxdim2 - Tmindim2 + 1:
            # Fits in both dimensions
            arrangement = np.zeros([NLdim1 * NLdim2])
            arrangement[0:n0] = initialstrength
            random.shuffle(arrangement)
            arrangement = np.reshape(arrangement, (NLdim1, NLdim2))
            Wpt[Currentiteration, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: int(
                rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) + 1,
            int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: int(
                rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) + 1, rdim1,
            rdim2] = arrangement
        else:
            # Fits in dim1 but not dim2
            arrangement = np.zeros([(Tmaxdim2 - int(rdim2 * ((NTdim2 - NLdim2) / NRdim2))) * NLdim1])
            arrangement[0:n0] = initialstrength
            random.shuffle(arrangement)
            arrangement = np.reshape(arrangement, (NLdim1, Tmaxdim2 - int(rdim2 * ((NTdim2 - NLdim2) / NRdim2))))
            Wpt[Currentiteration, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: int(
                rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) + 1,
            int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: Tmaxdim2 + 1, rdim1,
            rdim2] = arrangement
    elif int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= Tmaxdim2 - Tmindim2 + 1:
        # Doesn't fit into dim1 but fits into dim2
        arrangement = np.zeros([(Tmaxdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1))) * NLdim2])
        arrangement[0:n0] = initialstrength
        random.shuffle(arrangement)
        arrangement = np.reshape(arrangement, (Tmaxdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)), NLdim2))
        Wpt[Currentiteration, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: Tmaxdim1 + 1,
        int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) + 1,
        rdim1,
        rdim2] = arrangement
    else:
        # Doesn't fit into either dimension
        arrangement = np.zeros([(Tmaxdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1))) * (
            Tmaxdim2 - int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)))])
        arrangement[0:n0] = initialstrength
        random.shuffle(arrangement)
        arrangement = np.reshape(arrangement, (
            Tmaxdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)),
            Tmaxdim2 - int(rdim2 * ((NTdim2 - NLdim2) / NRdim2))))
        Wpt[Currentiteration, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: Tmaxdim1 + 1,
        int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: Tmaxdim2 + 1,
        rdim1,
        rdim2] = arrangement


def initialconnections():
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            connections(rdim1, rdim2)


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

            # Count connections
            synapses = np.array(np.nonzero(Wpt[Currentiteration - 1, :, :, rdim1, rdim2]))
            connections[rdim1, rdim2] = len(synapses[0, :])

            # Calculate similarity
            for synapse in range(int(connections[rdim1, rdim2])):
                Spt[synapses[0, synapse], synapses[1, synapse], rdim1, rdim2] = sum(
                    np.minimum(NormalisedCpm[:, rdim1, rdim2],
                               NormalisedCtm[:, synapses[0, synapse], synapses[1, synapse]]))

            # Calculate mean similarity
            totalSp[rdim1, rdim2] = sum(sum(Spt[:, :, rdim1, rdim2]))
            meanSp[rdim1, rdim2] = (totalSp[rdim1, rdim2] / connections[rdim1, rdim2]) - k

            # Calculate deltaW
            deltaWpt[Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1, rdim1, rdim2] = h * (
                Spt[Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1, rdim1, rdim2] - meanSp[rdim1, rdim2])

            # Calculate deltaWsum
            for synapse in range(int(connections[rdim1, rdim2])):
                deltaWsum[rdim1, rdim2] += deltaWpt[synapses[0, synapse], synapses[1, synapse], rdim1, rdim2]

            # Update Weight
            Wpt[Currentiteration, :, :, rdim1, rdim2] = ((Wpt[Currentiteration - 1, :, :, rdim1, rdim2] +
                                                          deltaWpt[:, :, rdim1, rdim2]) * Wtot[rdim1, rdim2]) / (
                                                            Wtot[rdim1, rdim2] + deltaWsum[rdim1, rdim2])


def removesynapses():
    synapses = np.array(np.nonzero(Wpt[Currentiteration, :, :, :, :]))
    for synapse in range(int(len(synapses[0, :]))):
        tdim1 = synapses[0, synapse]
        tdim2 = synapses[1, synapse]
        rdim1 = synapses[2, synapse]
        rdim2 = synapses[3, synapse]
        if Wpt[Currentiteration, tdim1, tdim2, rdim1, rdim2] < elim * Wtot[rdim1, rdim2]:
            Wpt[Currentiteration, tdim1, tdim2, rdim1, rdim2] = 0.


def addsynapses():
    synapses = np.array(
        np.nonzero(Wpt[Currentiteration, :, :, :, :]))

    for synapse in range(int(len(synapses[0, :]))):
        tdim1 = synapses[0, synapse]
        tdim2 = synapses[1, synapse]
        rdim1 = synapses[2, synapse]
        rdim2 = synapses[3, synapse]
        if Wpt[Currentiteration, tdim1, tdim2, rdim1, rdim2] > sprout * Wtot[rdim1, rdim2]:

            if Wpt[Currentiteration, tdim1 + 1, tdim2, rdim1, rdim2] == 0 and tdim1 + 1 <= Tmaxdim1:
                Wpt[Currentiteration, tdim1 + 1, tdim2, rdim1, rdim2] = newW * Wtot[rdim1, rdim2]
            if Wpt[Currentiteration, tdim1 - 1, tdim2, rdim1, rdim2] == 0 and tdim1 - 1 >= Tmindim1:
                Wpt[Currentiteration, tdim1 - 1, tdim2, rdim1, rdim2] = newW * Wtot[rdim1, rdim2]
            if Wpt[Currentiteration, tdim1, tdim2 + 1, rdim1, rdim2] == 0 and tdim2 + 1 <= Tmaxdim2:
                Wpt[Currentiteration, tdim1, tdim2 + 1, rdim1, rdim2] = newW * Wtot[rdim1, rdim2]
            if Wpt[Currentiteration, tdim1, tdim2 - 1, rdim1, rdim2] == 0 and tdim2 - 1 >= Tmindim2:
                Wpt[Currentiteration, tdim1, tdim2 - 1, rdim1, rdim2] = newW * Wtot[rdim1, rdim2]


def updatexFieldcentres():
    for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
        for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
            xFieldcentres[0, Currentiteration, tdim1, tdim2] = (Rmaxdim1 - Rmindim1 + 1) * tdim1 / (
                Tmaxdim1 - Tmindim1 + 1)
            xFieldcentres[1, Currentiteration, tdim1, tdim2] = (Rmaxdim2 - Rmindim2 + 1) * tdim2 / (
                Tmaxdim2 - Tmindim2 + 1)


def growretina():
    global Rmindim1, Rmaxdim1, Rmindim2, Rmaxdim2, Tmindim1, Tmaxdim1, Tmindim2, Tmaxdim2
    if Currentiteration % dstep == 0 and Currentiteration != 0:
        # Old map
        oldmap = np.zeros([NRdim1 + 1, NRdim2 + 1])
        oldmap[Rmindim1:Rmaxdim1 + 1, Rmindim2:Rmaxdim2 + 1] = 1

        # Grow retina
        if Rmindim1 > 1:
            Rmindim1 -= 1
        if Rmaxdim1 < NRdim1:
            Rmaxdim1 += 1
        if Rmindim2 > 1:
            Rmindim2 -= 1
        if Rmaxdim2 < NRdim2:
            Rmaxdim2 += 1

        # New map
        newmap = np.zeros([NRdim1 + 1, NRdim2 + 1])
        newmap[Rmindim1:Rmaxdim1 + 1, Rmindim2:Rmaxdim2 + 1] = 1

        # New connections
        for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
            for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
                if newmap[rdim1, rdim2] - oldmap[rdim1, rdim2] > 0:
                    connections(rdim1, rdim2)


def growtectum():
    global Rmindim1, Rmaxdim1, Rmindim2, Rmaxdim2, Tmindim1, Tmaxdim1, Tmindim2, Tmaxdim2
    if Currentiteration % dstep == 0 and Currentiteration != 0:
        if Tmaxdim1 < NTdim1 - 1:
            Tmaxdim1 += 2
        if Tmaxdim2 < NTdim2 - 1:
            Tmaxdim2 += 2
