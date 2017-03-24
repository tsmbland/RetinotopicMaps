import numpy as np
import Parameters as p
import os

################### VARIABLES ###################

Wpt = np.zeros(
    [p.Iterations / p.TRout + 1, p.NTdim1 + 2, p.NTdim2 + 2, p.NRdim1 + 2, p.NRdim2 + 2])  # synaptic strength matrix
Spt = np.zeros([p.NTdim1 + 2, p.NTdim2 + 2, p.NRdim1 + 2, p.NRdim2 + 2])  # similarity
Dpt = np.zeros([p.NTdim1 + 2, p.NTdim2 + 2, p.NRdim1 + 2, p.NRdim2 + 2])  # distance
Cra = np.zeros([p.NRdim1 + 2, p.NRdim2 + 2])
Crb = np.zeros([p.NRdim1 + 2, p.NRdim2 + 2])  # concentration of EphrinA/B in a retinal cell
Cta = np.zeros([p.Iterations / p.TRout + 1, p.NTdim1 + 2, p.NTdim2 + 2])
Ctb = np.zeros([p.Iterations / p.TRout + 1, p.NTdim1 + 2, p.NTdim2 + 2])  # concentration of EphrinA/B in a tectal cell
Ita = np.zeros([p.NTdim1 + 2, p.NTdim2 + 2])
Itb = np.zeros([p.NTdim1 + 2, p.NTdim2 + 2])  # induced label in a tectal cell
Nct = np.zeros([p.NTdim1 + 2, p.NTdim2 + 2])  # neighbour count for a tectal cell

xFieldcentres = np.zeros([2, p.Iterations + 1, p.NTdim1 + 1, p.NTdim2 + 1])  # expected field centres for tectal cells


#################### MODEL TYPE ####################

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

def setRetinalGradients():
    # Dim1
    if p.ynRdim1 != 0:
        aRdim1 = ((p.ymRdim1 - p.y0Rdim1) ** 2) / (p.ynRdim1 - 2 * p.ymRdim1 + p.y0Rdim1)
        bRdim1 = np.log((p.ynRdim1 - p.y0Rdim1) / aRdim1 + 1) / p.NRdim1
        cRdim1 = p.y0Rdim1 - aRdim1

        for rdim1 in range(1, p.NRdim1 + 1):
            Cra[rdim1, 1:p.NTdim2 + 1] = aRdim1 * np.exp(bRdim1 * rdim1) + cRdim1

    # Dim2
    if p.ynRdim2 != 0:
        aRdim2 = ((p.ymRdim2 - p.y0Rdim2) ** 2) / (p.ynRdim2 - 2 * p.ymRdim2 + p.y0Rdim2)
        bRdim2 = np.log((p.ynRdim2 - p.y0Rdim2) / aRdim2 + 1) / p.NRdim2
        cRdim2 = p.y0Rdim2 - aRdim2

        for rdim2 in range(1, p.NRdim2 + 1):
            Crb[1:p.NTdim1 + 1, rdim2] = aRdim2 * np.exp(bRdim2 * rdim2) + cRdim2

    # Add stochasticity
    for rdim1 in range(1, p.NRdim1 + 1):
        for rdim2 in range(1, p.NRdim2 + 1):
            Cra[rdim1, rdim2] = Cra[rdim1, rdim2] * (1 + np.random.uniform(-p.stochR, p.stochR))
            Crb[rdim1, rdim2] = Crb[rdim1, rdim2] * (1 + np.random.uniform(-p.stochR, p.stochR))


def setTectalGradients():
    # Dim1
    if p.ynTdim1 != 0:
        aTdim1 = ((p.ymTdim1 - p.y0Tdim1) ** 2) / (p.ynTdim1 - 2 * p.ymTdim1 + p.y0Tdim1)
        bTdim1 = np.log((p.ynTdim1 - p.y0Tdim1) / aTdim1 + 1) / p.NTdim1
        cTdim1 = p.y0Tdim1 - aTdim1

        for tdim1 in range(1, p.NTdim1 + 1):
            Cta[0, tdim1, 1:p.NTdim2 + 1] = aTdim1 * np.exp(bTdim1 * tdim1) + cTdim1

    # Dim2
    if p.ynTdim2 != 0:
        aTdim2 = ((p.ymTdim2 - p.y0Tdim2) ** 2) / (p.ynTdim2 - 2 * p.ymTdim2 + p.y0Tdim2)
        bTdim2 = np.log((p.ynTdim2 - p.y0Tdim2) / aTdim2 + 1) / p.NTdim2
        cTdim2 = p.y0Tdim2 - aTdim2

        for tdim2 in range(1, p.NTdim2 + 1):
            Ctb[0, 1:p.NTdim1 + 1, tdim2] = aTdim2 * np.exp(bTdim2 * tdim2) + cTdim2

    # Change gradient strength
    Cta[0, :, :] *= p.yLT
    Ctb[0, :, :] *= p.yLT

    # Add stochasticity
    for rdim1 in range(1, p.NRdim1 + 1):
        for rdim2 in range(1, p.NRdim2 + 1):
            Cta[0, rdim1, rdim2] = Cta[0, rdim1, rdim2] * (1 + np.random.uniform(-p.stochT, p.stochT))
            Ctb[0, rdim1, rdim2] = Ctb[0, rdim1, rdim2] * (1 + np.random.uniform(-p.stochT, p.stochT))


def updateNct():
    # Neuron map
    nm = np.zeros([p.NTdim1 + 2, p.NTdim2 + 2])
    nm[Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1] = 1

    # Neighbour Count
    for dim1 in range(Tmindim1, Tmaxdim1 + 1):
        for dim2 in range(Tmindim2, Tmaxdim2 + 1):
            Nct[dim1, dim2] = nm[dim1 + 1, dim2] + nm[dim1 - 1, dim2] + nm[dim1, dim2 + 1] + nm[dim1, dim2 - 1]


def updateI():
    Ita[:, :] = 0
    Itb[:, :] = 0
    wtotal = np.zeros([p.NTdim1 + 2, p.NTdim2 + 2])

    for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
        for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
            wtotal[tdim1, tdim2] = sum(sum(Wpt[Timepoint, tdim1, tdim2, :, :]))
            Ita[tdim1, tdim2] = sum(sum(Wpt[Timepoint, tdim1, tdim2, :, :] * Cra[:, :])) / wtotal[
                tdim1, tdim2]
            Itb[tdim1, tdim2] = sum(sum(Wpt[Timepoint, tdim1, tdim2, :, :] * Crb[:, :])) / wtotal[
                tdim1, tdim2]
    Ita[:, :] = np.nan_to_num(Ita[:, :])
    Itb[:, :] = np.nan_to_num(Itb[:, :])


def updateCta():
    for t in range(p.tc):
        for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
            for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                neighbourmean = (
                    (Cta[Timepoint, tdim1 + 1, tdim2] + Cta[Timepoint, tdim1 - 1, tdim2] + Cta[
                        Timepoint, tdim1, tdim2 + 1] + Cta[Timepoint, tdim1, tdim2 - 1]) / Nct[
                        tdim1, tdim2])

                Cta[Timepoint, tdim1, tdim2] += (p.alpha * (
                    1 - Ita[tdim1, tdim2] * Cta[Timepoint, tdim1, tdim2]) + p.beta * (
                                                     neighbourmean - Cta[
                                                         Timepoint, tdim1, tdim2])) * p.deltatc


def updateCtb():
    for t in range(p.tc):
        for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
            for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                neighbourmean = (
                    (Ctb[Timepoint, tdim1 + 1, tdim2] + Ctb[Timepoint, tdim1 - 1, tdim2] + Ctb[
                        Timepoint, tdim1, tdim2 + 1] + Ctb[Timepoint, tdim1, tdim2 - 1]) / Nct[
                        tdim1, tdim2])

                Ctb[Timepoint, tdim1, tdim2] += (p.alpha * (
                    Itb[tdim1, tdim2] - Ctb[Timepoint, tdim1, tdim2]) + p.beta * (
                                                     neighbourmean - Ctb[
                                                         Timepoint, tdim1, tdim2])) * p.deltatc


####################### SYNAPTIC MODIFICATION ###################

def connections(rdim1, rdim2):
    for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
        for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
            Wpt[Timepoint, tdim1, tdim2, rdim1, rdim2] = np.random.uniform(0, 0.0001)


def initialconnections():
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            connections(rdim1, rdim2)


def updateDpt():
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
                for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                    Dpt[tdim1, tdim2, rdim1, rdim2] = p.distA * (
                        (Cra[rdim1, rdim2] * Cta[Timepoint, tdim1, tdim2] - 1) ** 2) + p.distB * (
                        (Crb[rdim1, rdim2] - Ctb[Timepoint, tdim1, tdim2]) ** 2)


def updateSpt():
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
                for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                    Spt[tdim1, tdim2, rdim1, rdim2] = np.exp(-Dpt[tdim1, tdim2, rdim1, rdim2] / (2 * p.kappa ** 2))


def updateWpt():
    for t in range(p.tw):
        numerator = Wpt[Timepoint, :, :, :, :] + p.deltatw * p.gamma * Spt
        denominator = np.zeros([p.NRdim1 + 2, p.NRdim2 + 2])
        for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
            for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
                denominator[rdim1, rdim2] = sum(
                    sum((Wpt[Timepoint, :, :, rdim1, rdim2] + p.deltatw * p.gamma * Spt[
                                                                                    :, :,
                                                                                    rdim1,
                                                                                    rdim2])))
                Wpt[Timepoint, :, :, rdim1, rdim2] = numerator[:, :, rdim1, rdim2] / denominator[rdim1, rdim2]


def removesynapses():
    Wpt[Wpt < 0.001] = 0


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
        Cta[Timepoint, :, :] = Cta[Timepoint - 1, :, :]
        Ctb[Timepoint, :, :] = Ctb[Timepoint - 1, :, :]
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
    np.save('../../../RetinotopicMapsData/%s/EphrinA' % ('{0:04}'.format(JobID)), Cta)
    np.save('../../../RetinotopicMapsData/%s/EphrinB' % ('{0:04}'.format(JobID)), Ctb)
    np.save('../../../RetinotopicMapsData/%s/xFieldCentres' % ('{0:04}'.format(JobID)), xFieldcentres)
    np.save('../../../RetinotopicMapsData/%s/PrimaryTR' % ('{0:04}'.format(JobID)), p.TRout)
