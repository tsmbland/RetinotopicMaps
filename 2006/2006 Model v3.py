# Fixed tectal gradients

import numpy as np
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

# Synaptic modification
gamma = 0.1
kappa = 0.0504
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

Currentiteration = 0


####################### FUNCTIONS ###################

def initialconnections():
    for rdim1 in range(Rmindim1, Rmaxdim1 + 1):
        for rdim2 in range(Rmindim2, Rmaxdim2 + 1):
            for tdim1 in range(Tmindim1, Tmaxdim1 + 1):
                for tdim2 in range(Tmindim2, Tmaxdim2 + 1):
                    Wpt[0, tdim1, tdim2, rdim1, rdim2] = np.random.uniform(0, 0.0001)


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
            Cta[:, tdim1, 1:NTdim2 + 1] = aTdim1 * np.exp(bTdim1 * tdim1) + cTdim1

    # Dim2
    if ynTdim2 != 0:
        aTdim2 = ((ymTdim2 - y0Tdim2) ** 2) / (ynTdim2 - 2 * ymTdim2 + y0Tdim2)
        bTdim2 = np.log((ynTdim2 - y0Tdim2) / aTdim2 + 1) / NTdim2
        cTdim2 = y0Tdim2 - aTdim2

        for tdim2 in range(1, NTdim2 + 1):
            Ctb[:, 1:NTdim1 + 1, tdim2] = aTdim2 * np.exp(bTdim2 * tdim2) + cTdim2


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


def updateWpt():
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


######################## ALGORITM #######################


# Set Gradients
setRetinalGradients()
setTectalGradients()

# Initial Connections
initialconnections()

# Iterations
for iteration in range(Iterations):
    Currentiteration += 1

    updateDpt()
    updateSpt()
    updateWpt()

    sys.stdout.write('\r%i percent' % (iteration * 100 / Iterations))
    sys.stdout.flush()

#################### EXPORT DATA #################

np.save('../Temporary Data/Weightmatrix', Wpt[0:Iterations + 2:TRout, :, :, :, :])
np.save('../Temporary Data/EphrinA', Cta[0:Iterations + 2:TRout, :, :])
np.save('../Temporary Data/EphrinB', Ctb[0:Iterations + 2:TRout, :, :])

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
