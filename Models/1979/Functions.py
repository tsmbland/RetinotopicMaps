import numpy as np
import random
import Parameters as p
import os

################### VARIABLES ###################

M = p.Mdim1 * p.Mdim2

Wpt = np.zeros([p.Iterations / p.TRout + 1, p.NTdim1 + 2, p.NTdim2 + 2, p.NRdim1 + 2, p.NRdim2 + 2])
Wtot = np.zeros([p.NRdim1 + 1, p.NRdim2 + 1])

Qpm = np.zeros([M, p.NRdim1 + 2, p.NRdim2 + 2])
Qtm = np.zeros([M, p.NTdim1 + 2, p.NTdim2 + 2])

Cpm = np.zeros([M, p.NRdim1 + 2, p.NRdim2 + 2])
Ctm = np.zeros([M, p.Iterations / p.TRout + 1, p.NTdim1 + 2, p.NTdim2 + 2])
NormalisedCpm = np.zeros([M, p.NRdim1 + 2, p.NRdim2 + 2])
NormalisedCtm = np.zeros([M, p.NTdim1 + 2, p.NTdim2 + 2])

NCp = np.zeros([p.NRdim1 + 2, p.NRdim2 + 2])
NCt = np.zeros([p.NTdim1 + 2, p.NTdim2 + 2])

xFieldcentres = np.zeros([2, p.Iterations + 1, p.NTdim1 + 1, p.NTdim2 + 1])


################## MODEL TYPE ###################

def typestandard():
    global Rmindim1, Rmaxdim1, Rmindim2, Rmaxdim2, Tmindim1, Tmaxdim1, Tmindim2, Tmaxdim2, Currentiteration, Timepoint
    Rmindim1 = 1
    Rmaxdim1 = p.NRdim1
    Rmindim2 = 1
    Rmaxdim2 = p.NRdim2
    Tmindim1 = 1
    Tmaxdim1 = p.NTdim1
    Tmindim2 = 1
    Tmaxdim2 = p.NTdim2
    Currentiteration = 0
    Timepoint = 0


def typemismatchsurgery():
    global Rmindim1, Rmaxdim1, Rmindim2, Rmaxdim2, Tmindim1, Tmaxdim1, Tmindim2, Tmaxdim2, Currentiteration, Timepoint
    Rmindim1 = p.sRmindim1
    Rmaxdim1 = p.sRmaxdim1
    Rmindim2 = p.sRmindim2
    Rmaxdim2 = p.sRmaxdim2
    Tmindim1 = p.sTmindim1
    Tmaxdim1 = p.sTmaxdim1
    Tmindim2 = p.sTmindim2
    Tmaxdim2 = p.sTmaxdim2
    Currentiteration = 0
    Timepoint = 0


def typedevelopment():
    global Rmindim1, Rmaxdim1, Rmindim2, Rmaxdim2, Tmindim1, Tmaxdim1, Tmindim2, Tmaxdim2, Currentiteration, Timepoint
    Rmindim1 = p.dRmindim1
    Rmaxdim1 = p.dRmaxdim1
    Rmindim2 = p.dRmindim2
    Rmaxdim2 = p.dRmaxdim2
    Tmindim1 = p.dTmindim1
    Tmaxdim1 = p.dTmaxdim1
    Tmindim2 = p.dTmindim2
    Tmaxdim2 = p.dTmaxdim2
    Currentiteration = 0
    Timepoint = 0


################### CONCENTRATIONS ###################

def setmarkerlocations():
    if p.Mdim1 > 1:
        markerspacingdim1 = p.NRdim1 / (p.Mdim1 - 1)
    else:
        markerspacingdim1 = 0
    if p.Mdim2 > 1:
        markerspacingdim2 = p.NRdim2 / (p.Mdim2 - 1)
    else:
        markerspacingdim2 = 0

    m = 0
    locationdim1 = 1
    locationdim2 = 1
    for mdim2 in range(p.Mdim2 - 1):
        for mdim1 in range(p.Mdim1 - 1):
            Qpm[m, locationdim1, locationdim2] = p.Q
            locationdim1 += markerspacingdim1
            m += 1
        Qpm[m, p.NRdim1, locationdim2] = p.Q
        locationdim1 = 1
        locationdim2 += markerspacingdim2
        m += 1

    for mdim1 in range(p.Mdim1 - 1):
        Qpm[m, locationdim1, p.NRdim2] = p.Q
        locationdim1 += markerspacingdim1
        m += 1
    Qpm[m, p.NRdim1, p.NRdim2] = p.Q


def updateNc():
    # Presynaptic neuron map
    nmp = np.zeros([p.NRdim1 + 2, p.NRdim2 + 2])
    nmp[Rmindim1:Rmaxdim1 + 1, Rmindim2:Rmaxdim2 + 1] = 1

    # Tectal neuron map
    nmt = np.zeros([p.NTdim1 + 2, p.NTdim2 + 2])
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
    while averagemarkerchange > p.stab:
        concchange = np.zeros([M, len(Cpm[0, :, 0]), len(Cpm[0, 0, :])])
        for m in range(M):
            for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
                for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
                    concchange[m, rdim1, rdim2] = (-p.a * Cpm[m, rdim1, rdim2] + p.d * (
                        Cpm[m, rdim1, rdim2 + 1] + Cpm[m, rdim1, rdim2 - 1] + Cpm[m, rdim1 + 1, rdim2] +
                        Cpm[m, rdim1 - 1, rdim2] - NCp[rdim1, rdim2] * Cpm[m, rdim1, rdim2]) + Qpm[
                                                       m, rdim1, rdim2])

        averagemarkerchange = (sum(sum(sum(concchange))) / sum(sum(sum(Cpm)))) * 100
        Cpm[:, :, :] += (concchange * p.deltat)


def updateretinalconcs():
    for t in range(p.tc):
        for m in range(M):
            for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
                for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
                    Cpm[m, rdim1, rdim2] += (-p.a * Cpm[m, rdim1, rdim2] + p.d * (
                        Cpm[m, rdim1, rdim2 + 1] + Cpm[m, rdim1, rdim2 - 1] + Cpm[
                            m, rdim1 + 1, rdim2] +
                        Cpm[m, rdim1 - 1, rdim2] - NCp[rdim1, rdim2] * Cpm[
                            m, rdim1, rdim2]) + Qpm[m, rdim1, rdim2]) * p.deltat


def updatetectalconcs():
    for t in range(p.tc):
        for m in range(M):
            for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
                for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                    Ctm[m, Timepoint, tdim1, tdim2] += (-p.a * Ctm[m, Timepoint, tdim1, tdim2] + p.d * (
                        Ctm[m, Timepoint, tdim1, tdim2 + 1] + Ctm[m, Timepoint, tdim1, tdim2 - 1] + Ctm[
                            m, Timepoint, tdim1 + 1, tdim2] +
                        Ctm[m, Timepoint, tdim1 - 1, tdim2] - NCt[tdim1, tdim2] * Ctm[
                            m, Timepoint, tdim1, tdim2]) + Qtm[
                                                            m, tdim1, tdim2]) * p.deltat


def normaliseCpm():
    NormalisedCpm[:, :, :] = 0
    markersum = np.sum(Cpm, axis=0)
    for m in range(M):
        for dim1 in range(Rmindim1, Rmaxdim1 + 1):
            for dim2 in range(Rmindim2, Rmaxdim2 + 1):
                NormalisedCpm[m, dim1, dim2] = Cpm[m, dim1, dim2] / markersum[dim1, dim2]
                if NormalisedCpm[m, dim1, dim2] < p.E:
                    NormalisedCpm[m, dim1, dim2] = 0
    NormalisedCpm[:, :, :] = np.nan_to_num(NormalisedCpm[:, :, :])


def normaliseCtm():
    NormalisedCtm[:, :, :] = 0
    markersum = np.sum(Ctm[:, Timepoint, :, :], axis=0)
    for m in range(M):
        for dim1 in range(Tmindim1, Tmaxdim1 + 1):
            for dim2 in range(Tmindim2, Tmaxdim2 + 1):
                NormalisedCtm[m, dim1, dim2] = Ctm[m, Timepoint, dim1, dim2] / markersum[dim1, dim2]
                if NormalisedCtm[m, dim1, dim2] < p.E:
                    NormalisedCtm[m, dim1, dim2] = 0
    NormalisedCtm[:, :, :] = np.nan_to_num(NormalisedCtm[:, :, :])


##################### SYNAPTIC MODIFICATION ###################

def setWtot():
    Wtot[1:p.NRdim1 + 1, 1:p.NRdim2 + 1] = p.Wmax / p.td
    Wtot[Rmindim1:Rmaxdim1 + 1, Rmindim2:Rmaxdim2 + 1] = p.Wmax


def updateWtot():
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            if Wtot[rdim1, rdim2] < p.Wmax:
                Wtot[rdim1, rdim2] += p.Wmax / p.td


def connections(rdim1, rdim2):
    initialstrength = Wtot[rdim1, rdim2] / p.n0
    if int(rdim1 * ((p.NTdim1 - p.NLdim1) / p.NRdim1) + p.NLdim1) <= Tmaxdim1 - Tmindim1 + 1:
        if int(rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2) + p.NLdim2) <= Tmaxdim2 - Tmindim2 + 1:
            # Fits in both dimensions
            arrangement = np.zeros([p.NLdim1 * p.NLdim2])
            arrangement[0:p.n0] = initialstrength
            random.shuffle(arrangement)
            arrangement = np.reshape(arrangement, (p.NLdim1, p.NLdim2))
            Wpt[Timepoint, int(rdim1 * ((p.NTdim1 - p.NLdim1) / p.NRdim1)) + 1: int(
                rdim1 * ((p.NTdim1 - p.NLdim1) / p.NRdim1) + p.NLdim1) + 1,
            int(rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2)) + 1: int(
                rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2) + p.NLdim2) + 1, rdim1,
            rdim2] = arrangement
        else:
            # Fits in dim1 but not dim2
            arrangement = np.zeros([(Tmaxdim2 - int(rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2))) * p.NLdim1])
            arrangement[0:p.n0] = initialstrength
            random.shuffle(arrangement)
            arrangement = np.reshape(arrangement,
                                     (p.NLdim1, Tmaxdim2 - int(rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2))))
            Wpt[Timepoint, int(rdim1 * ((p.NTdim1 - p.NLdim1) / p.NRdim1)) + 1: int(
                rdim1 * ((p.NTdim1 - p.NLdim1) / p.NRdim1) + p.NLdim1) + 1,
            int(rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2)) + 1: Tmaxdim2 + 1, rdim1,
            rdim2] = arrangement
    elif int(rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2) + p.NLdim2) <= Tmaxdim2 - Tmindim2 + 1:
        # Doesn't fit into dim1 but fits into dim2
        arrangement = np.zeros([(Tmaxdim1 - int(rdim1 * ((p.NTdim1 - p.NLdim1) / p.NRdim1))) * p.NLdim2])
        arrangement[0:p.n0] = initialstrength
        random.shuffle(arrangement)
        arrangement = np.reshape(arrangement, (Tmaxdim1 - int(rdim1 * ((p.NTdim1 - p.NLdim1) / p.NRdim1)), p.NLdim2))
        Wpt[Timepoint, int(rdim1 * ((p.NTdim1 - p.NLdim1) / p.NRdim1)) + 1: Tmaxdim1 + 1,
        int(rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2)) + 1: int(
            rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2) + p.NLdim2) + 1,
        rdim1,
        rdim2] = arrangement
    else:
        # Doesn't fit into either dimension
        arrangement = np.zeros([(Tmaxdim1 - int(rdim1 * ((p.NTdim1 - p.NLdim1) / p.NRdim1))) * (
            Tmaxdim2 - int(rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2)))])
        arrangement[0:p.n0] = initialstrength
        random.shuffle(arrangement)
        arrangement = np.reshape(arrangement, (
            Tmaxdim1 - int(rdim1 * ((p.NTdim1 - p.NLdim1) / p.NRdim1)),
            Tmaxdim2 - int(rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2))))
        Wpt[Timepoint, int(rdim1 * ((p.NTdim1 - p.NLdim1) / p.NRdim1)) + 1: Tmaxdim1 + 1,
        int(rdim2 * ((p.NTdim2 - p.NLdim2) / p.NRdim2)) + 1: Tmaxdim2 + 1,
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
                                                                        Timepoint, tdim1, tdim2, :, :]))


def updateWeight():
    Spt = np.zeros([p.NTdim1 + 2, p.NTdim2 + 2, p.NRdim1 + 2, p.NRdim2 + 2])
    deltaWpt = np.zeros([p.NTdim1 + 2, p.NTdim2 + 2, p.NRdim1 + 2, p.NRdim2 + 2])
    totalSp = np.zeros([p.NRdim1 + 2, p.NRdim2 + 2])
    meanSp = np.zeros([p.NRdim1 + 2, p.NRdim2 + 2])
    deltaWsum = np.zeros([p.NRdim1 + 2, p.NRdim2 + 2])
    connections = np.zeros([p.NRdim1 + 2, p.NRdim2 + 2])

    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):

            # Count connections
            synapses = np.array(np.nonzero(Wpt[Timepoint, :, :, rdim1, rdim2]))
            connections[rdim1, rdim2] = len(synapses[0, :])

            # Calculate similarity
            for synapse in range(int(connections[rdim1, rdim2])):
                Spt[synapses[0, synapse], synapses[1, synapse], rdim1, rdim2] = sum(
                    np.minimum(NormalisedCpm[:, rdim1, rdim2],
                               NormalisedCtm[:, synapses[0, synapse], synapses[1, synapse]]))

            # Calculate mean similarity
            totalSp[rdim1, rdim2] = sum(sum(Spt[:, :, rdim1, rdim2]))
            meanSp[rdim1, rdim2] = (totalSp[rdim1, rdim2] / connections[rdim1, rdim2]) - p.k

            # Calculate deltaW
            deltaWpt[Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1, rdim1, rdim2] = p.h * (
                Spt[Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1, rdim1, rdim2] - meanSp[rdim1, rdim2])

            # Calculate deltaWsum
            for synapse in range(int(connections[rdim1, rdim2])):
                deltaWsum[rdim1, rdim2] += deltaWpt[synapses[0, synapse], synapses[1, synapse], rdim1, rdim2]

            # Update Weight
            Wpt[Timepoint, :, :, rdim1, rdim2] += (deltaWpt[:, :, rdim1, rdim2] * Wtot[rdim1, rdim2]) / (
                Wtot[rdim1, rdim2] + deltaWsum[rdim1, rdim2])


def removesynapses():
    synapses = np.array(np.nonzero(Wpt[Timepoint, :, :, :, :]))
    for synapse in range(int(len(synapses[0, :]))):
        tdim1 = synapses[0, synapse]
        tdim2 = synapses[1, synapse]
        rdim1 = synapses[2, synapse]
        rdim2 = synapses[3, synapse]
        if Wpt[Timepoint, tdim1, tdim2, rdim1, rdim2] < p.elim * Wtot[rdim1, rdim2]:
            Wpt[Timepoint, tdim1, tdim2, rdim1, rdim2] = 0.


def addsynapses():
    synapses = np.array(
        np.nonzero(Wpt[Timepoint, :, :, :, :]))

    for synapse in range(int(len(synapses[0, :]))):
        tdim1 = synapses[0, synapse]
        tdim2 = synapses[1, synapse]
        rdim1 = synapses[2, synapse]
        rdim2 = synapses[3, synapse]
        if Wpt[Timepoint, tdim1, tdim2, rdim1, rdim2] > p.sprout * Wtot[rdim1, rdim2]:

            if Wpt[Timepoint, tdim1 + 1, tdim2, rdim1, rdim2] == 0 and tdim1 + 1 <= Tmaxdim1:
                Wpt[Timepoint, tdim1 + 1, tdim2, rdim1, rdim2] = p.newW * Wtot[rdim1, rdim2]
            if Wpt[Timepoint, tdim1 - 1, tdim2, rdim1, rdim2] == 0 and tdim1 - 1 >= Tmindim1:
                Wpt[Timepoint, tdim1 - 1, tdim2, rdim1, rdim2] = p.newW * Wtot[rdim1, rdim2]
            if Wpt[Timepoint, tdim1, tdim2 + 1, rdim1, rdim2] == 0 and tdim2 + 1 <= Tmaxdim2:
                Wpt[Timepoint, tdim1, tdim2 + 1, rdim1, rdim2] = p.newW * Wtot[rdim1, rdim2]
            if Wpt[Timepoint, tdim1, tdim2 - 1, rdim1, rdim2] == 0 and tdim2 - 1 >= Tmindim2:
                Wpt[Timepoint, tdim1, tdim2 - 1, rdim1, rdim2] = p.newW * Wtot[rdim1, rdim2]


######################### GROWTH ########################

def growretina():
    global Rmindim1, Rmaxdim1, Rmindim2, Rmaxdim2, Tmindim1, Tmaxdim1, Tmindim2, Tmaxdim2
    if Currentiteration % p.dstep == 0 and Currentiteration != 0:
        # Old map
        oldmap = np.zeros([p.NRdim1 + 1, p.NRdim2 + 1])
        oldmap[Rmindim1:Rmaxdim1 + 1, Rmindim2:Rmaxdim2 + 1] = 1

        # Grow retina
        if Rmindim1 > 1:
            Rmindim1 -= 1
        if Rmaxdim1 < p.NRdim1:
            Rmaxdim1 += 1
        if Rmindim2 > 1:
            Rmindim2 -= 1
        if Rmaxdim2 < p.NRdim2:
            Rmaxdim2 += 1

        # New map
        newmap = np.zeros([p.NRdim1 + 1, p.NRdim2 + 1])
        newmap[Rmindim1:Rmaxdim1 + 1, Rmindim2:Rmaxdim2 + 1] = 1

        # New connections
        for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
            for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
                if newmap[rdim1, rdim2] - oldmap[rdim1, rdim2] > 0:
                    connections(rdim1, rdim2)


def growtectum():
    global Rmindim1, Rmaxdim1, Rmindim2, Rmaxdim2, Tmindim1, Tmaxdim1, Tmindim2, Tmaxdim2
    if Currentiteration % p.dstep == 0 and Currentiteration != 0:
        if Tmaxdim1 < p.NTdim1 - 1:
            Tmaxdim1 += 2
        if Tmaxdim2 < p.NTdim2 - 1:
            Tmaxdim2 += 2


########################### MISC #########################

def updatetimepoint():
    global Timepoint, Currentiteration
    Currentiteration += 1
    if (Currentiteration - 1) % p.TRout == 0:
        Timepoint += 1
        Ctm[:, Timepoint, :, :] = Ctm[:, Timepoint - 1, :, :]
        Wpt[Timepoint, :, :, :, :] = Wpt[Timepoint - 1, :, :, :, :]


def updatexFieldcentres():
    for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
        for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
            xFieldcentres[0, Timepoint, tdim1, tdim2] = (Rmaxdim1 - Rmindim1 + 1) * tdim1 / (
                Tmaxdim1 - Tmindim1 + 1)
            xFieldcentres[1, Timepoint, tdim1, tdim2] = (Rmaxdim2 - Rmindim2 + 1) * tdim2 / (
                Tmaxdim2 - Tmindim2 + 1)


def savedata(JobID):
    if not os.path.isdir('../../../RetinotopicMapsData/%s' % ('{0:04}'.format(JobID))):
        os.makedirs('../../../RetinotopicMapsData/%s' % ('{0:04}'.format(JobID)))
    np.save('../../../RetinotopicMapsData/%s/Weightmatrix' % ('{0:04}'.format(JobID)), Wpt)
    np.save('../../../RetinotopicMapsData/%s/TectalConcentrations' % ('{0:04}'.format(JobID)), Ctm)
    np.save('../../../RetinotopicMapsData/%s/xFieldCentres' % ('{0:04}'.format(JobID)), xFieldcentres)
    np.save('../../../RetinotopicMapsData/%s/PrimaryTR' % ('{0:04}'.format(JobID)), p.TRout)
