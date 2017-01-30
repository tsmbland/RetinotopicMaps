import numpy as np
import time
import sys

start = time.time()

#################### PARAMETERS #####################

# General
Iterations = 100  # number of weight iterations
NRdim1 = 10  # initial number of retinal cells
NRdim2 = 10
NTdim1 = 10  # initial number of tectal cells
NTdim2 = 10

# Retinal Gradients
y0Rdim1 = 0.5  # conc in cell 0
ymRdim1 = 1.5  # conc in cell NRdim1
ynRdim1 = 3.5  # conc in cell NRdim1/2
y0Rdim2 = 0.5
ymRdim2 = 1.5
ynRdim2 = 3.5

# Tectal Gradients
y0Tdim1 = 0.5  # conc in cell 0
ymTdim1 = 1.5  # conc in cell NTdim1
ynTdim1 = 3.5  # conc in cell NTdim1/2
y0Tdim2 = 0.5
ymTdim2 = 1.5
ynTdim2 = 3.5

# Tectal concentrations
a = 0.006  # decay constant
d = 0.01  # diffusion length constant
alpha = 0.005
beta = 0.01
deltatc = 0.1  # deltaC time step
tc = 1  # concentration iterations per iteration

# Synaptic modification
W = 1  # total strength available to each presynaptic fibre
gamma = 0.1
kappa = 0.72
elim = 0.005  # elimination threshold
deltatw = 0.1  # deltaW time step
tw = 1  # weight iterations per iteration


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

Wpt = np.zeros([Iterations*tw + 1, NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])  # synaptic strength matrix
Spt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])  # similarity
Cra = np.zeros([NRdim1 + 2, NRdim2 + 2])
Crb = np.zeros([NRdim1 + 2, NRdim2 + 2])  # concentration of EphrinA/B in a retinal cell
Cta = np.zeros([NTdim1 + 2, NTdim2 + 2])
Ctb = np.zeros([NTdim1 + 2, NTdim2 + 2])  # concentration of EphrinA/B in a tectal cell
Ita = np.zeros([NTdim1 + 2, NTdim2 + 2])
Itb = np.zeros([NTdim1 + 2, NTdim2 + 2])  # induced label in a tectal cell

Nct = np.zeros([NTdim1 + 2, NTdim2 + 2])  # neighbour count for a tectal cell


####################### FUNCTIONS ###################

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
            Cta[tdim1, 1:NTdim2 + 1] = aTdim1 * np.exp(bTdim1 * tdim1) + cTdim1

    # Dim2
    if ynTdim2 != 0:
        aTdim2 = ((ymTdim2 - y0Tdim2) ** 2) / (ynTdim2 - 2 * ymTdim2 + y0Tdim2)
        bTdim2 = np.log((ynTdim2 - y0Tdim2) / aTdim2 + 1) / NTdim2
        cTdim2 = y0Tdim2 - aTdim2

        for tdim2 in range(1, NTdim2 + 1):
            Ctb[1:NTdim1 + 1, tdim2] = aTdim2 * np.exp(bTdim2 * tdim2) + cTdim2


def updateNct():
    # Neuron map
    nm = np.zeros([NTdim1 + 2, NTdim2 + 2])
    nm[tmindim1:tmaxdim1 + 1, tmindim2:tmaxdim2 + 1] = 1

    # Neighbour Count
    for dim1 in range(tmindim1, tmaxdim1 + 1):
        for dim2 in range(tmindim2, tmaxdim2 + 1):
            Nct[dim1, dim2] = nm[dim1 + 1, dim2] + nm[dim1 - 1, dim2] + nm[dim1, dim2 + 1] + nm[dim1, dim2 - 1]


def updateI():
    Ita[:, :] = 0
    Itb[:, :] = 0
    wtotal = np.zeros([NTdim1 + 2, NTdim2 + 2])

    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            wtotal[tdim1, tdim2] = sum(sum(Wpt[currentiteration, tdim1, tdim2, :, :]))
            Ita[tdim1, tdim2] = sum(sum(Wpt[currentiteration, tdim1, tdim2, :, :] * Cra[:, :])) / wtotal[tdim1, tdim2]
            Itb[tdim1, tdim2] = sum(sum(Wpt[currentiteration, tdim1, tdim2, :, :] * Crb[:, :])) / wtotal[tdim1, tdim2]


def concchangeCta():
    deltacta = np.zeros([NTdim1 + 2, NTdim2 + 2])
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            deltacta[tdim1, tdim2] = alpha * (1 - Ita[tdim1, tdim2] * Cta[tdim1, tdim2])
    return deltacta


def concchangeCtb():
    deltactb = np.zeros([NTdim1 + 2, NTdim2 + 2])
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            deltactb[tdim1, tdim2] = alpha * (Itb[tdim1, tdim2] - Ctb[tdim1, tdim2])
    return deltactb


def updateSpt():
    dist = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    for rdim1 in range(rmindim1, rmaxdim1 + 1):
        for rdim2 in range(rmindim2, rmaxdim2 + 1):
            for tdim1 in range(tmindim1, tmaxdim1 + 1):
                for tdim2 in range(tmindim2, tmaxdim2 + 1):
                    # Calculate distance
                    dist[tdim1, tdim2, rdim1, rdim2] = ((Cra[rdim1, rdim2] * Cta[tdim1, tdim2] - 1) ** 2) + ((Crb[
                                                                                                                  rdim1, rdim2] -
                                                                                                              Ctb[
                                                                                                                  tdim1, tdim2]) ** 2)

                    # Calculate similarity
                    Spt[tdim1, tdim2, rdim1, rdim2] = np.exp(-dist[tdim1, tdim2, rdim1, rdim2] / (2 * kappa ** 2))


def weightchange():
    deltaWpt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    denominator = np.zeros([NRdim1 + 2, NRdim2 + 2])
    numerator = Wpt[currentiteration - 1, :, :, :, :] + deltatw * gamma * Spt

    for rdim1 in range(rmindim1, rmaxdim1 + 1):
        for rdim2 in range(rmindim2, rmaxdim2 + 1):
            denominator[rdim1, rdim2] = sum(sum((Wpt[currentiteration - 1, :, :, rdim1, rdim2] + deltatw * gamma * Spt[
                                                                                                                   :, :,
                                                                                                                   rdim1,
                                                                                                                   rdim2])))
            deltaWpt[:, :, rdim1, rdim2] = numerator[:, :, rdim1, rdim2] / denominator[rdim1, rdim2]

    deltaWpt -= Wpt[currentiteration - 1, :, :, :, :]
    return deltaWpt


######################## ALGORITM #######################
currentiteration = 0

# Set Gradients
setRetinalGradients()
setTectalGradients()

# Initial Connections


for rdim1 in range(rmindim1, rmaxdim1 + 1):
    for rdim2 in range(rmindim2, rmaxdim2 + 1):
        for tdim1 in range(tmindim1, tmaxdim1 + 1):
            for tdim2 in range(tmindim2, tmaxdim2 + 1):
                Wpt[0, tdim1, tdim2, rdim1, rdim2] = np.random.uniform(0, 0.0001)

# Initial Tectal Concentrations
updateNct()
updateI()
for t in range(tc):
    deltaCta = concchangeCta()
    deltaCtb = concchangeCtb()
    Cta += (deltaCta * deltatc)
    Ctb += (deltaCtb * deltatc)

# Iterations
for iteration in range(Iterations):

    # Weight Change
    updateSpt()
    for t in range(tw):
        currentiteration += 1
        deltaW = weightchange()
        Wpt[currentiteration, :, :, :, :] = Wpt[currentiteration - 1, :, :, :, :] + (deltatw * deltaW)

    # Concentration Change
    updateI()
    for t in range(tc):
        deltaCta = concchangeCta()
        deltaCtb = concchangeCtb()
        Cta += (deltaCta * deltatc)
        Ctb += (deltaCtb * deltatc)

    sys.stdout.write('\r%i percent' % (iteration * 100 / Iterations))
    sys.stdout.flush()

#################### EXPORT DATA #################

np.save('../Temporary Data/Weightmatrix', Wpt)

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
