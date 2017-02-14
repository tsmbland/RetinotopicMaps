import numpy as np
import random

#################### PARAMETERS #####################

# General
Iterations = 100  # number of weight iterations
NRdim1 = 20  # initial number of retinal cells
NRdim2 = 20
NTdim1 = 20  # initial number of tectal cells
NTdim2 = 20

# Establishment of initial contacts
n0 = 10  # number of initial random contact
NLdim1 = 20  # sets initial bias
NLdim2 = 20

# Retinal Gradients
y0Rdim1 = 1.0  # conc in cell 0
ymRdim1 = 2.0  # conc in cell NRdim1/2
ynRdim1 = 3.5  # conc in cell NRdim1
y0Rdim2 = 0.1
ymRdim2 = 0.5
ynRdim2 = 1.0

# Tectal Gradients
y0Tdim1 = 1.0  # conc in cell 0
ymTdim1 = 0.5  # conc in cell NTdim1/2
ynTdim1 = 0.3  # conc in cell NTdim1
y0Tdim2 = 0.1
ymTdim2 = 0.5
ynTdim2 = 1.0

# Tectal concentrations
alpha = 0.05
beta = 0.05
deltatc = 1  # deltaC time step
tc = 1  # concentration iterations per iteration

# Synaptic modification
W = 1  # total strength available to each presynaptic fibre
gamma = 0.1
kappa = 0.0504
elim = 0.005  # elimination threshold
newW = 0.01  # weight of new synapses
sprout = 0.02  # sprouting threshold
deltatw = 1  # deltaW time step
tw = 1  # weight iterations per iteration

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

Wpt = np.zeros([Iterations + 1, NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])  # synaptic strength matrix
Spt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])  # similarity
Dpt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])  # distance
Cra = np.zeros([NRdim1 + 2, NRdim2 + 2])
Crb = np.zeros([NRdim1 + 2, NRdim2 + 2])  # concentration of EphrinA/B in a retinal cell
Cta = np.zeros([Iterations + 1, NTdim1 + 2, NTdim2 + 2])
Ctb = np.zeros([Iterations + 1, NTdim1 + 2, NTdim2 + 2])  # concentration of EphrinA/B in a tectal cell
Ita = np.zeros([NTdim1 + 2, NTdim2 + 2])
Itb = np.zeros([NTdim1 + 2, NTdim2 + 2])  # induced label in a tectal cell
Nct = np.zeros([NTdim1 + 2, NTdim2 + 2])  # neighbour count for a tectal cell

Currentiteration = 0


####################### FUNCTIONS ###################

def initialconnections1():
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
                for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                    Wpt[0, tdim1, tdim2, rdim1, rdim2] = np.random.uniform(0, 0.0001)


def initialconections2(rdim1, rdim2):
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
            wtotal[tdim1, tdim2] = sum(sum(Wpt[Currentiteration - 1, tdim1, tdim2, :, :]))
            Ita[tdim1, tdim2] = sum(sum(Wpt[Currentiteration - 1, tdim1, tdim2, :, :] * Cra[:, :])) / wtotal[
                tdim1, tdim2]
            Itb[tdim1, tdim2] = sum(sum(Wpt[Currentiteration - 1, tdim1, tdim2, :, :] * Crb[:, :])) / wtotal[
                tdim1, tdim2]
    Ita[:, :] = np.nan_to_num(Ita[:, :])
    Itb[:, :] = np.nan_to_num(Itb[:, :])


def updateCta():
    Cta[Currentiteration, :, :] = Cta[Currentiteration - 1, :, :]
    for t in range(tc):
        for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
            for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                neighbourmean = (
                    (Cta[Currentiteration, tdim1 + 1, tdim2] + Cta[Currentiteration, tdim1 - 1, tdim2] + Cta[
                        Currentiteration, tdim1, tdim2 + 1] + Cta[Currentiteration, tdim1, tdim2 - 1]) / Nct[
                        tdim1, tdim2])

                Cta[Currentiteration, tdim1, tdim2] += (alpha * (
                    1 - Ita[tdim1, tdim2] * Cta[Currentiteration, tdim1, tdim2]) + beta * (
                                                            neighbourmean - Cta[
                                                                Currentiteration, tdim1, tdim2])) * deltatc


def updateCtb():
    Ctb[Currentiteration, :, :] = Ctb[Currentiteration - 1, :, :]
    for t in range(tc):
        for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
            for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                neighbourmean = (
                    (Ctb[Currentiteration, tdim1 + 1, tdim2] + Ctb[Currentiteration, tdim1 - 1, tdim2] + Ctb[
                        Currentiteration, tdim1, tdim2 + 1] + Ctb[Currentiteration, tdim1, tdim2 - 1]) / Nct[
                        tdim1, tdim2])

                Ctb[Currentiteration, tdim1, tdim2] += (alpha * (
                    Itb[tdim1, tdim2] - Ctb[Currentiteration, tdim1, tdim2]) + beta * (
                                                            neighbourmean - Ctb[
                                                                Currentiteration, tdim1, tdim2])) * deltatc


def updateDpt():
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
                for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                    Dpt[tdim1, tdim2, rdim1, rdim2] = ((Cra[rdim1, rdim2] * Cta[
                        Currentiteration, tdim1, tdim2] - 1) ** 2) + (
                                                          (
                                                              Crb[rdim1, rdim2] - Ctb[
                                                                  Currentiteration, tdim1, tdim2]) ** 2)


def updateSpt():
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
                for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                    Spt[tdim1, tdim2, rdim1, rdim2] = np.exp(-Dpt[tdim1, tdim2, rdim1, rdim2] / (2 * kappa ** 2))


def updateWpt1():
    Wpt[Currentiteration, :, :, :, :] = Wpt[Currentiteration - 1, :, :, :, :]
    for t in range(tw):
        numerator = Wpt[Currentiteration - 1, :, :, :, :] + deltatw * gamma * Spt
        denominator = np.zeros([NRdim1 + 2, NRdim2 + 2])
        for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
            for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
                denominator[rdim1, rdim2] = sum(
                    sum((Wpt[Currentiteration - 1, :, :, rdim1, rdim2] + deltatw * gamma * Spt[
                                                                                                   :, :,
                                                                                                   rdim1,
                                                                                                   rdim2])))
                Wpt[Currentiteration, :, :, rdim1, rdim2] = numerator[:, :, rdim1, rdim2] / denominator[rdim1, rdim2]


def updateWpt2():
    Wpt[Currentiteration, :, :, :, :] = Wpt[Currentiteration - 1, :, :, :, :]
    synapses = np.array(np.nonzero(Wpt[Currentiteration, :, :, :, :]))

    # Calculate similarity
    dist = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    sim = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    for synapse in range(int(len(synapses[0, :]))):
        tdim1 = synapses[0, synapse]
        tdim2 = synapses[1, synapse]
        rdim1 = synapses[2, synapse]
        rdim2 = synapses[3, synapse]
        dist[tdim1, tdim2, rdim1, rdim2] = (
                                               (Cra[rdim1, rdim2] * Cta[
                                                   Currentiteration, tdim1, tdim2] - 1) ** 2) + (
                                               (Crb[rdim1, rdim2] - Ctb[Currentiteration, tdim1, tdim2]) ** 2)
        sim[tdim1, tdim2, rdim1, rdim2] = np.exp(-dist[tdim1, tdim2, rdim1, rdim2] / (2 * kappa ** 2))

    # Update weight
    for t in range(tw):
        numerator = W * (Wpt[Currentiteration, :, :, :, :] + deltatw * gamma * sim)
        denominator = np.zeros([NRdim1 + 2, NRdim2 + 2])
        for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
            for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
                denominator[rdim1, rdim2] = W + sum(sum(deltatw * 10000 * gamma * sim[:, :, rdim1, rdim2]))

        for synapse in range(int(len(synapses[0, :]))):
            tdim1 = synapses[0, synapse]
            tdim2 = synapses[1, synapse]
            rdim1 = synapses[2, synapse]
            rdim2 = synapses[3, synapse]
            Wpt[Currentiteration, tdim1, tdim2, rdim1, rdim2] = numerator[tdim1, tdim2, rdim1, rdim2] / denominator[
                rdim1, rdim2]


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
