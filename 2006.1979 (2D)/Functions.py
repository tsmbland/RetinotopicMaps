import numpy as np
import random

#################### PARAMETERS #####################

# General
Iterations = 500  # number of weight iterations
NRdim1 = 40  # initial number of retinal cells
NRdim2 = 40
NTdim1 = 40  # initial number of tectal cells
NTdim2 = 40

# Establishment of initial contacts
n0 = 10  # number of initial random contact
NLdim1 = 40  # sets initial bias
NLdim2 = 40

# Retinal Gradients
y0Rdim1 = 1.0  # conc in cell 0
ymRdim1 = 2.0  # conc in cell NRdim1/2
ynRdim1 = 3.5  # conc in cell NRdim1
y0Rdim2 = 0.1
ymRdim2 = 0.5
ynRdim2 = 1.0
stochR = 0.1

# Tectal Gradients
y0Tdim1 = 1 / 1.0  # conc in cell 0 (before multiplication by yLT)
ymTdim1 = 1 / 2.0  # conc in cell NTdim1/2
ynTdim1 = 1 / 3.5  # conc in cell NTdim1
y0Tdim2 = 0.1
ymTdim2 = 0.5
ynTdim2 = 1.0
yLT = 0.3
stochT = 0.1

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
dRmindim2 = 1
dRmaxdim2 = 1  # NRdim2 // 4
dTmindim1 = 1
dTmaxdim1 = NTdim1 // 2
dTmindim2 = 1
dTmaxdim2 = 1  # NTdim2 // 4
dstep = 30  # time between growth iterations
td = 300  # time taken for new fibres to gain full strength

# Tectal concentrations
alpha = 0.005
beta = 0.1
deltatc = 1  # deltaC time step
tc = 5  # concentration iterations per iteration

# Synaptic modification
Wmax = 1.  # total strength available to each presynaptic fibre
gamma = 0.01
kappa = 1
k = 0.001
elim = 0.005  # elimination threshold
newW = 0.01  # weight of new synapses
sprout = 0.02  # sprouting threshold
deltatw = 1  # deltaW time step
tw = 1  # weight iterations per iteration

# Output
TRout = 10  # temporal resoultion of output files

################### VARIABLES ###################

Wpt = np.zeros([Iterations / TRout + 1, NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])  # synaptic strength matrix
Wtot = np.zeros([NRdim1 + 1, NRdim2 + 1])  # total strength available to a fibre
Spt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])  # similarity
Dpt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])  # distance
Cra = np.zeros([NRdim1 + 2, NRdim2 + 2])
Crb = np.zeros([NRdim1 + 2, NRdim2 + 2])  # concentration of EphrinA/B in a retinal cell
Cta = np.zeros([Iterations / TRout + 1, NTdim1 + 2, NTdim2 + 2])
Ctb = np.zeros([Iterations / TRout + 1, NTdim1 + 2, NTdim2 + 2])  # concentration of EphrinA/B in a tectal cell
Ita = np.zeros([NTdim1 + 2, NTdim2 + 2])
Itb = np.zeros([NTdim1 + 2, NTdim2 + 2])  # induced label in a tectal cell
Nct = np.zeros([NTdim1 + 2, NTdim2 + 2])  # neighbour count for a tectal cell

xFieldcentres = np.zeros([2, Iterations + 1, NTdim1 + 1, NTdim2 + 1])  # expected field centres for tectal cells

Currentiteration = 0
Timepoint = 0


####################### FUNCTIONS ###################

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


def connections(rdim1, rdim2):
    initialstrength = Wtot[rdim1, rdim2] / n0
    if int(rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) <= Tmaxdim1 - Tmindim1 + 1:
        if int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= Tmaxdim2 - Tmindim2 + 1:
            # Fits in both dimensions
            arrangement = np.zeros([NLdim1 * NLdim2])
            arrangement[0:n0] = initialstrength
            random.shuffle(arrangement)
            arrangement = np.reshape(arrangement, (NLdim1, NLdim2))
            Wpt[Timepoint, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: int(
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
            Wpt[Timepoint, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: int(
                rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) + 1,
            int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: Tmaxdim2 + 1, rdim1,
            rdim2] = arrangement
    elif int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= Tmaxdim2 - Tmindim2 + 1:
        # Doesn't fit into dim1 but fits into dim2
        arrangement = np.zeros([(Tmaxdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1))) * NLdim2])
        arrangement[0:n0] = initialstrength
        random.shuffle(arrangement)
        arrangement = np.reshape(arrangement, (Tmaxdim1 - int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)), NLdim2))
        Wpt[Timepoint, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: Tmaxdim1 + 1,
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
        Wpt[Timepoint, int(rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: Tmaxdim1 + 1,
        int(rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: Tmaxdim2 + 1,
        rdim1,
        rdim2] = arrangement


def initialconnections():
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            connections(rdim1, rdim2)


def setRetinalGradients():
    # Dim1
    if ynRdim1 != 0:
        aRdim1 = ((ymRdim1 - y0Rdim1) ** 2) / (ynRdim1 - 2 * ymRdim1 + y0Rdim1)
        bRdim1 = np.log((ynRdim1 - y0Rdim1) / aRdim1 + 1) / NRdim1
        cRdim1 = y0Rdim1 - aRdim1

        for rdim1 in range(1, NRdim1 + 1):
            Cra[rdim1, 1:NTdim2 + 1] = aRdim1 * np.exp(bRdim1 * rdim1) + cRdim1

    # Dim2
    if ynRdim2 != 0:
        aRdim2 = ((ymRdim2 - y0Rdim2) ** 2) / (ynRdim2 - 2 * ymRdim2 + y0Rdim2)
        bRdim2 = np.log((ynRdim2 - y0Rdim2) / aRdim2 + 1) / NRdim2
        cRdim2 = y0Rdim2 - aRdim2

        for rdim2 in range(1, NRdim2 + 1):
            Crb[1:NTdim1 + 1, rdim2] = aRdim2 * np.exp(bRdim2 * rdim2) + cRdim2

    # Add stochasticity
    for rdim1 in range(1, NRdim1 + 1):
        for rdim2 in range(1, NRdim2 + 1):
            Cra[rdim1, rdim2] = Cra[rdim1, rdim2] * (1 + np.random.uniform(-stochR, stochR))
            Crb[rdim1, rdim2] = Crb[rdim1, rdim2] * (1 + np.random.uniform(-stochR, stochR))


def setTectalGradients():
    # Dim1
    if ynTdim1 != 0:
        aTdim1 = ((ymTdim1 - y0Tdim1) ** 2) / (ynTdim1 - 2 * ymTdim1 + y0Tdim1)
        bTdim1 = np.log((ynTdim1 - y0Tdim1) / aTdim1 + 1) / NTdim1
        cTdim1 = y0Tdim1 - aTdim1

        for tdim1 in range(1, NTdim1 + 1):
            Cta[0, tdim1, 1:NTdim2 + 1] = aTdim1 * np.exp(bTdim1 * tdim1) + cTdim1

    # Dim2
    if ynTdim2 != 0:
        aTdim2 = ((ymTdim2 - y0Tdim2) ** 2) / (ynTdim2 - 2 * ymTdim2 + y0Tdim2)
        bTdim2 = np.log((ynTdim2 - y0Tdim2) / aTdim2 + 1) / NTdim2
        cTdim2 = y0Tdim2 - aTdim2

        for tdim2 in range(1, NTdim2 + 1):
            Ctb[0, 1:NTdim1 + 1, tdim2] = aTdim2 * np.exp(bTdim2 * tdim2) + cTdim2

    # Change gradient strength
    Cta[0, :, :] *= yLT
    Ctb[0, :, :] *= yLT

    # Add stochasticity
    for rdim1 in range(1, NRdim1 + 1):
        for rdim2 in range(1, NRdim2 + 1):
            Cta[0, rdim1, rdim2] = Cta[0, rdim1, rdim2] * (1 + np.random.uniform(-stochT, stochT))
            Ctb[0, rdim1, rdim2] = Ctb[0, rdim1, rdim2] * (1 + np.random.uniform(-stochT, stochT))


def updateNct():
    # Neuron map
    nm = np.zeros([NTdim1 + 2, NTdim2 + 2])
    nm[Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1] = 1

    # Neighbour Count
    for dim1 in range(Tmindim1, Tmaxdim1 + 1):
        for dim2 in range(Tmindim2, Tmaxdim2 + 1):
            Nct[dim1, dim2] = nm[dim1 + 1, dim2] + nm[dim1 - 1, dim2] + nm[dim1, dim2 + 1] + nm[dim1, dim2 - 1]


def updateI():
    Ita[:, :] = 0
    Itb[:, :] = 0
    wtotal = np.zeros([NTdim1 + 2, NTdim2 + 2])

    for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
        for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
            wtotal[tdim1, tdim2] = sum(sum(Wpt[Timepoint, tdim1, tdim2, :, :]))
            Ita[tdim1, tdim2] = sum(sum(Wpt[Timepoint, tdim1, tdim2, :, :] * Cra[:, :])) / wtotal[
                tdim1, tdim2]
            Itb[tdim1, tdim2] = sum(sum(Wpt[Timepoint, tdim1, tdim2, :, :] * Crb[:, :])) / wtotal[
                tdim1, tdim2]
    Ita[:, :] = np.nan_to_num(Ita[:, :])
    Itb[:, :] = np.nan_to_num(Itb[:, :])


def updatetimepoint():
    global Timepoint, Currentiteration
    Currentiteration += 1
    if (Currentiteration - 1) % TRout == 0:
        Timepoint += 1
        Cta[Timepoint, :, :] = Cta[Timepoint - 1, :, :]
        Ctb[Timepoint, :, :] = Ctb[Timepoint - 1, :, :]
        Wpt[Timepoint, :, :, :, :] = Wpt[Timepoint - 1, :, :, :, :]


def updateCta():
    for t in range(tc):
        for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
            for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                neighbourmean = (
                    (Cta[Timepoint, tdim1 + 1, tdim2] + Cta[Timepoint, tdim1 - 1, tdim2] + Cta[
                        Timepoint, tdim1, tdim2 + 1] + Cta[Timepoint, tdim1, tdim2 - 1]) / Nct[
                        tdim1, tdim2])

                Cta[Timepoint, tdim1, tdim2] += (alpha * (
                    1 - Ita[tdim1, tdim2] * Cta[Timepoint, tdim1, tdim2]) + beta * (
                                                     neighbourmean - Cta[
                                                         Timepoint, tdim1, tdim2])) * deltatc


def updateCtb():
    for t in range(tc):
        for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
            for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                neighbourmean = (
                    (Ctb[Timepoint, tdim1 + 1, tdim2] + Ctb[Timepoint, tdim1 - 1, tdim2] + Ctb[
                        Timepoint, tdim1, tdim2 + 1] + Ctb[Timepoint, tdim1, tdim2 - 1]) / Nct[
                        tdim1, tdim2])

                Ctb[Timepoint, tdim1, tdim2] += (alpha * (
                    Itb[tdim1, tdim2] - Ctb[Timepoint, tdim1, tdim2]) + beta * (
                                                     neighbourmean - Ctb[
                                                         Timepoint, tdim1, tdim2])) * deltatc


def updateWpt():
    deltaWpt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    deltaWsum = np.zeros([NRdim1 + 2, NRdim2 + 2])
    dist = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    sim = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    totalsim = np.zeros([NRdim1 + 2, NRdim2 + 2])
    meansim = np.zeros([NRdim1 + 2, NRdim2 + 2])
    connections = np.zeros([NRdim1 + 2, NRdim2 + 2])

    # Similarity
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):

            # Count connections
            synapses = np.array(np.nonzero(Wpt[Timepoint, :, :, rdim1, rdim2]))
            connections[rdim1, rdim2] = len(synapses[0, :])

            # Calculate similarity
            for synapse in range(int(len(synapses[0, :]))):
                tdim1 = synapses[0, synapse]
                tdim2 = synapses[1, synapse]
                dist[tdim1, tdim2, rdim1, rdim2] = ((Crb[rdim1, rdim2] - Ctb[Timepoint, tdim1, tdim2]) ** 2) + (
                    (Cra[rdim1, rdim2] * Cta[Timepoint, tdim1, tdim2] - 1) ** 2)
                sim[tdim1, tdim2, rdim1, rdim2] = np.exp(-dist[tdim1, tdim2, rdim1, rdim2] / (2 * kappa ** 2))

            # Calculate mean similarity
            totalsim[rdim1, rdim2] = sum(sum(sim[:, :, rdim1, rdim2]))
            meansim[rdim1, rdim2] = (totalsim[rdim1, rdim2] / connections[rdim1, rdim2]) - k

    # Update weight
    for t in range(tw):
        for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
            for rdim2 in range(Rmindim2, Rmaxdim2 + 1):

                # Calculate deltaW
                deltaWpt[Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1, rdim1, rdim2] = deltatw * gamma * (
                    sim[Tmindim1:Tmaxdim1 + 1, Tmindim2:Tmaxdim2 + 1, rdim1, rdim2] - meansim[rdim1, rdim2])

                # Calculate deltaWsum
                synapses = np.array(np.nonzero(Wpt[Timepoint, :, :, rdim1, rdim2]))
                for synapse in range(int(len(synapses[0, :]))):
                    deltaWsum[rdim1, rdim2] += deltaWpt[synapses[0, synapse], synapses[1, synapse], rdim1, rdim2]

                # Update Weight
                for synapse in range(int(len(synapses[0, :]))):
                    tdim1 = synapses[0, synapse]
                    tdim2 = synapses[1, synapse]
                    Wpt[Timepoint, tdim1, tdim2, rdim1, rdim2] = (Wpt[
                                                                      Timepoint, tdim1, tdim2, rdim1, rdim2] +
                                                                  deltaWpt[tdim1, tdim2, rdim1, rdim2]) * Wtot[
                                                                     rdim1, rdim2] / (
                                                                     Wtot[rdim1, rdim2] + deltaWsum[
                                                                         rdim1, rdim2])


def removesynapses():
    synapses = np.array(np.nonzero(Wpt[Timepoint, :, :, :, :]))
    for synapse in range(int(len(synapses[0, :]))):
        tdim1 = synapses[0, synapse]
        tdim2 = synapses[1, synapse]
        rdim1 = synapses[2, synapse]
        rdim2 = synapses[3, synapse]
        if Wpt[Timepoint, tdim1, tdim2, rdim1, rdim2] < elim * Wtot[rdim1, rdim2]:
            Wpt[Timepoint, tdim1, tdim2, rdim1, rdim2] = 0.


def addsynapses():
    synapses = np.array(
        np.nonzero(Wpt[Timepoint, :, :, :, :]))

    for synapse in range(int(len(synapses[0, :]))):
        tdim1 = synapses[0, synapse]
        tdim2 = synapses[1, synapse]
        rdim1 = synapses[2, synapse]
        rdim2 = synapses[3, synapse]
        if Wpt[Timepoint, tdim1, tdim2, rdim1, rdim2] > sprout * Wtot[rdim1, rdim2]:

            if Wpt[Timepoint, tdim1 + 1, tdim2, rdim1, rdim2] == 0 and tdim1 + 1 <= Tmaxdim1:
                Wpt[Timepoint, tdim1 + 1, tdim2, rdim1, rdim2] = newW * Wtot[rdim1, rdim2]
            if Wpt[Timepoint, tdim1 - 1, tdim2, rdim1, rdim2] == 0 and tdim1 - 1 >= Tmindim1:
                Wpt[Timepoint, tdim1 - 1, tdim2, rdim1, rdim2] = newW * Wtot[rdim1, rdim2]
            if Wpt[Timepoint, tdim1, tdim2 + 1, rdim1, rdim2] == 0 and tdim2 + 1 <= Tmaxdim2:
                Wpt[Timepoint, tdim1, tdim2 + 1, rdim1, rdim2] = newW * Wtot[rdim1, rdim2]
            if Wpt[Timepoint, tdim1, tdim2 - 1, rdim1, rdim2] == 0 and tdim2 - 1 >= Tmindim2:
                Wpt[Timepoint, tdim1, tdim2 - 1, rdim1, rdim2] = newW * Wtot[rdim1, rdim2]


def updatexFieldcentres():
    for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
        for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
            xFieldcentres[0, Timepoint, tdim1, tdim2] = (Rmaxdim1 - Rmindim1 + 1) * tdim1 / (
                Tmaxdim1 - Tmindim1 + 1)
            xFieldcentres[1, Timepoint, tdim1, tdim2] = (Rmaxdim2 - Rmindim2 + 1) * tdim2 / (
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
