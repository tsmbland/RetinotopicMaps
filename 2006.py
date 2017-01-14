import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys

start = time.time()

#################### PARAMETERS #####################

# Layer Dimensions
NRdim1 = 10  # initial number of retinal cells
NRdim2 = 10
NTdim1 = 10  # initial number of tectal cells
NTdim2 = 10

# Retinal Concentrations
ephAmax = 2  # max retinal concentration of EphrinA
ephBmax = 2
ephAgrad = 7  # steepness of retinal Ephrin gradient (lower = steeper)
ephBgrad = 7

# Establishment of initial contacts
n0 = 7  # number of initial random contact
NLdim1 = 7  # sets initial bias
NLdim2 = 7

# Tectal concentrations
a = 0.006  # decay constant
d = 0.01  # diffusion length constant
alpha = 0.005
beta = 0.01
deltatc = 0.1  # deltaC time step
tc = 50  # concentration iterations per iteration

# Synaptic modification
W = 1  # total strength available to each presynaptic fibre
gamma = 0.1
kappa = 0.72
elim = 0.005  # elimination threshold
newW = 0.01  # weight of new synapses
deltatw = 1  # deltaW time step
tw = 1  # weight iterations per iteration
Iterations = 100  # number of weight iterations

# Plot
Rplotdim = 2  # retina dimension plotted (1 or 2)
Rplotslice = NRdim2 // 2  # slice location in the other dimension
Tplotdim = 2
Tplotslice = NTdim2 // 2
Precisiontstep = 10  # time step for precision value update

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

Wpt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])  # synaptic strength matrix
Spt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])  # similarity
Cra = np.zeros([NRdim1 + 2, NRdim2 + 2])
Crb = np.zeros([NRdim1 + 2, NRdim2 + 2])  # concentration of EphrinA/B in a retinal cell
Cta = np.zeros([NTdim1 + 2, NTdim2 + 2])
Ctb = np.zeros([NTdim1 + 2, NTdim2 + 2])  # concentration of EphrinA/B in a tectal cell
Ita = np.zeros([NTdim1 + 2, NTdim2 + 2])
Itb = np.zeros([NTdim1 + 2, NTdim2 + 2])  # induced label in a tectal cell

Nt = np.zeros([NTdim1 + 2, NTdim2 + 2])  # neighbour count for a tectal cell

Fieldcentre = np.zeros([2, NTdim1 + 2, NTdim2 + 2])
Fieldseparation = []
Fieldsize = []
Orientation = []
Time = []


################## RETINA #####################

# RETINAL EPHRIN GRADIENTS

def set_retinal_gradients():
    # EphA gradient in dim1
    for rdim2 in range(1, NRdim2 + 1):
        for rdim1 in range(1, NRdim1 + 1):
            Cra[rdim1, rdim2] = np.exp((NRdim1 / (NRdim1 * ephAgrad)) * (rdim1 - NRdim1)) * ephAmax

    # EphB gradient in dim2
    for rdim1 in range(1, NRdim2 + 1):
        for rdim2 in range(1, NRdim1 + 1):
            Crb[rdim1, rdim2] = np.exp((NRdim2 / (NRdim2 * ephBgrad)) * (rdim2 - NRdim2)) * ephBmax


set_retinal_gradients()


################# INITIAL CONNECTIONS ###################

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


for rdim1 in range(rmindim1, rmaxdim1 + 1):
    for rdim2 in range(rmindim2, rmaxdim2 + 1):
        initialconections(rdim1, rdim2)


#################### TECTUM #####################

# INITIAL TECTAL CONCENTRATIONS

# Update neighbour count (only needed when layer size changes)
def update_Nt():
    # Neuron map
    nm = np.zeros([NTdim1 + 2, NTdim2 + 2])
    nm[tmindim1:tmaxdim1 + 1, tmindim2:tmaxdim2 + 1] = 1

    # Neighbour Count
    for dim1 in range(tmindim1, tmaxdim1 + 1):
        for dim2 in range(tmindim2, tmaxdim2 + 1):
            Nt[dim1, dim2] = nm[dim1 + 1, dim2] + nm[dim1 - 1, dim2] + nm[dim1, dim2 + 1] + nm[dim1, dim2 - 1]


# Calculate induced label
def updateI():
    Ita[:, :] = 0
    Itb[:, :] = 0
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            a = 0
            b = 0
            wtotal = 0
            for rdim1 in range(rmindim1, rmaxdim1 + 1):
                for rdim2 in range(rmindim2, rmaxdim2 + 1):
                    a += Wpt[tdim1, tdim2, rdim1, rdim2] * Cra[rdim1, rdim2]
                    b += Wpt[tdim1, tdim2, rdim1, rdim2] * Crb[rdim1, rdim2]
                    wtotal += Wpt[tdim1, tdim2, rdim1, rdim2]
            if wtotal != 0:
                Ita[tdim1, tdim2] = a / wtotal
                Itb[tdim1, tdim2] = b / wtotal


def concchangeCta():
    deltacta = np.zeros([NTdim1 + 2, NTdim2 + 2])
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            deltacta[tdim1, tdim2] = alpha * (Ita[tdim1, tdim2] - Cta[tdim1, tdim2]) + d * (
                Cta[tdim1, tdim2 + 1] + Cta[tdim1, tdim2 - 1] + Cta[tdim1 + 1, tdim2] + Cta[tdim1 - 1, tdim2] - Nt[
                    tdim1, tdim2] * Cta[tdim1, tdim2])
    return deltacta


def concchangeCtb():
    deltactb = np.zeros([NTdim1 + 2, NTdim2 + 2])
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            deltactb[tdim1, tdim2] = alpha * (Itb[tdim1, tdim2] - Ctb[tdim1, tdim2]) + d * (
                Ctb[tdim1, tdim2 + 1] + Ctb[tdim1, tdim2 - 1] + Ctb[tdim1 + 1, tdim2] + Ctb[tdim1 - 1, tdim2] - Nt[
                    tdim1, tdim2] * Ctb[tdim1, tdim2])
    return deltactb


update_Nt()
updateI()
for t in range(tc):
    deltaCta = concchangeCta()
    deltaCtb = concchangeCtb()
    Cta += (deltaCta * deltatc)
    Ctb += (deltaCtb * deltatc)


################## PRECISION MEASURES ################

def field_centre():
    fieldcentre = np.zeros([2, NTdim1 + 2, NTdim2 + 2])
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            totaldim1 = 0
            totaldim2 = 0
            weightsumdim1 = 0
            weightsumdim2 = 0

            for rdim1 in range(rmindim1, rmaxdim1 + 1):
                for rdim2 in range(rmindim2, rmaxdim2 + 1):
                    totaldim1 += rdim1 * Wpt[tdim1, tdim2, rdim1, rdim2]
                    weightsumdim1 += Wpt[tdim1, tdim2, rdim1, rdim2]
                    totaldim2 += rdim2 * Wpt[tdim1, tdim2, rdim1, rdim2]
                    weightsumdim2 += Wpt[tdim1, tdim2, rdim1, rdim2]

            if totaldim1 != 0:
                fieldcentre[0, tdim1, tdim2] = totaldim1 / weightsumdim1
            if totaldim2 != 0:
                fieldcentre[1, tdim1, tdim2] = totaldim2 / weightsumdim2

    return fieldcentre


def orientation():
    # Gives mean distance between actual field centre and expected field centre
    totaldistance = 0
    count = 0

    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            if Fieldcentre[0, tdim1, tdim2] != 0 and Fieldcentre[1, tdim1, tdim2] != 0:
                totaldistance += np.sqrt((Fieldcentre[0, tdim1, tdim2] - (nRdim1 * tdim1 / nTdim1)) ** 2 + (
                    Fieldcentre[1, tdim1, tdim2] - (nRdim2 * tdim2 / nTdim2)) ** 2)
                count += 1

    meandistance = totaldistance / count

    return meandistance


def field_separation():
    totaldistance = 0
    count = 0

    # Field distance with closest neighbours in dim2
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            if Fieldcentre[0, tdim1, tdim2] != 0 and Fieldcentre[1, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentre[0, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentre[1, tdim1, tdim2])
        for fieldcell in range(len(fieldlistdim1) - 1):
            totaldistance += np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2)
            count += 1

    # Field distance with closest neighbours in dim1
    for tdim2 in range(tmindim2, tmaxdim2 + 1):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim1 in range(tmindim1, tmaxdim1 + 1):
            if Fieldcentre[0, tdim1, tdim2] != 0 and Fieldcentre[1, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentre[0, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentre[1, tdim1, tdim2])
        for fieldcell in range(len(fieldlistdim1) - 1):
            totaldistance += np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2)
            count += 1

    meanseparation = totaldistance / count
    return meanseparation


def field_size():
    totalarea = 0
    count = 0
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):

            area = 0
            # Scanning in dim1
            for rdim2 in range(rmindim2, rmaxdim2 + 1):
                width = 0
                rdim1 = 0
                weight = 0
                while weight == 0 and rdim1 < nRdim1:
                    weight = Wpt[tdim1, tdim2, rdim1, rdim2]
                    rdim1 += 1
                min = rdim1 - 1
                if weight != 0:
                    rdim1 = rmaxdim1
                    weight = 0
                    while weight == 0:
                        weight = Wpt[tdim1, tdim2, rdim1, rdim2]
                        rdim1 -= 1
                    max = rdim1 + 1
                    width = max - min
                area += width

            # Scanning in dim2
            for rdim1 in range(rmindim1, rmaxdim1 + 1):
                width = 0
                rdim2 = 0
                weight = 0
                while weight == 0 and rdim2 < nRdim2:
                    weight = Wpt[tdim1, tdim2, rdim1, rdim2]
                    rdim2 += 1
                min = rdim2 - 1
                if weight != 0:
                    rdim2 = rmaxdim2
                    weight = 0
                    while weight == 0:
                        weight = Wpt[tdim1, tdim2, rdim1, rdim2]
                        rdim2 -= 1
                    max = rdim2 + 1
                    width = max - min
                area += width

            # Field size estimation
            totalarea += area / 2
            count += 1

    # Mean field size estimation
    meanarea = totalarea / count
    return meanarea


##################### WEIGHT CHANGE ####################

def updateSpt():
    for rdim1 in range(rmindim1, rmaxdim1 + 1):
        for rdim2 in range(rmindim2, rmaxdim2 + 1):
            for tdim1 in range(tmindim1, tmaxdim1 + 1):
                for tdim2 in range(tmindim2, tmaxdim2 + 1):
                    # Calculate distance
                    dist = (Cra[rdim1, rdim2] - Cta[tdim1, tdim2]) ** 2 + (Crb[rdim1, rdim2] - Ctb[
                        tdim1, tdim2]) ** 2

                    # Calculate similarity
                    Spt[tdim1, tdim2, rdim1, rdim2] = np.exp(-dist / (2 * kappa ** 2))


def weightchange():
    # SYNAPTIC WEIGHT
    deltaWpt = np.zeros([NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2])
    for rdim1 in range(rmindim1, rmaxdim1 + 1):
        for rdim2 in range(rmindim2, rmaxdim2 + 1):
            Sptsum = 0
            for tdim1 in range(tmindim1, tmaxdim1 + 1):
                for tdim2 in range(tmindim2, tmaxdim2 + 1):
                    # Calculate Sptsum
                    Sptsum += (Wpt[tdim1, tdim2, rdim1, rdim2] + gamma * Spt[tdim1, tdim2, rdim1, rdim2])
            for tdim1 in range(tmindim1, tmaxdim1 + 1):
                for tdim2 in range(tmindim2, tmaxdim2 + 1):
                    if Wpt[tdim1, tdim2, rdim1, rdim2] != 0:
                        # Calculate deltaW
                        deltaWpt[tdim1, tdim2, rdim1, rdim2] = ((Wpt[tdim1, tdim2, rdim1, rdim2] + gamma * Spt[
                            tdim1, tdim2, rdim1, rdim2]) / Sptsum) - Wpt[tdim1, tdim2, rdim1, rdim2]

    return deltaWpt


def removesynapses():
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            for rdim1 in range(rmindim1, rmaxdim1 + 1):
                for rdim2 in range(rmindim2, rmaxdim2 + 1):
                    if Wpt[tdim1, tdim2, rdim1, rdim2] < elim * W:
                        Wpt[tdim1, tdim2, rdim1, rdim2] = 0


def addsynapses():
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            for rdim1 in range(rmindim1, rmaxdim1 + 1):
                for rdim2 in range(rmindim2, rmaxdim2 + 1):
                    if Wpt[tdim1, tdim2, rdim1, rdim2] == 0 and (
                                            Wpt[tdim1 + 1, tdim2, rdim1, rdim2] > 0.02 * W or Wpt[
                                            tdim1 - 1, tdim2, rdim1, rdim2] > 0.02 * W or Wpt[
                                    tdim1, tdim2 + 1, rdim1, rdim2] > 0.02 * W or Wpt[
                                tdim1, tdim2 - 1, rdim1, rdim2] > 0.02 * W):
                        Wpt[tdim1, tdim2, rdim1, rdim2] = newW * W


###################### ITERATIONS #######################

for iteration in range(Iterations):

    updateSpt()

    for t in range(tw):
        deltaW = weightchange()
        Wpt += (deltatw * deltaW)
    removesynapses()
    addsynapses()

    updateI()

    for t in range(tc):
        deltaCta = concchangeCta()
        deltaCtb = concchangeCtb()
        Cta += (deltaCta * deltatc)
        Ctb += (deltaCtb * deltatc)

    if iteration % Precisiontstep == 0 or iteration == Iterations or iteration == 1:
        Fieldcentre = field_centre()
        Fieldseparation.append(field_separation())
        Fieldsize.append(field_size())
        Orientation.append(orientation())
        Time.append(iteration)

    sys.stdout.write('\r%i percent' % (iteration * 100 / Iterations))
    sys.stdout.flush()

##################### PLOTS #######################
if Rplotdim == 1:
    rplotmindim1 = rplotmin = rmindim1
    rplotmaxdim1 = rplotmax = rmaxdim1
    rplotmindim2 = Rplotslice
    rplotmaxdim2 = Rplotslice
elif Rplotdim == 2:
    rplotmindim1 = Rplotslice
    rplotmaxdim1 = Rplotslice
    rplotmindim2 = rplotmin = rmindim2
    rplotmaxdim2 = rplotmax = rmaxdim2
if Tplotdim == 1:
    tplotmindim1 = tplotmin = tmindim1
    tplotmaxdim1 = tplotmax = tmaxdim1
    tplotmindim2 = Tplotslice
    tplotmaxdim2 = Tplotslice
elif Tplotdim == 2:
    tplotmindim1 = Tplotslice
    tplotmaxdim1 = Tplotslice
    tplotmindim2 = tplotmin = tmindim2
    tplotmaxdim2 = tplotmax = tmaxdim2


def tabulate_weight_matrix():
    table = np.zeros([(rplotmax - rplotmin + 1) * (tplotmax - tplotmin + 1), 6])
    row = 0
    deltaw = weightchange()
    for rdim1 in range(rplotmindim1, rplotmaxdim1 + 1):
        for rdim2 in range(rplotmindim2, rplotmaxdim2 + 1):
            for tdim1 in range(tplotmindim1, tplotmaxdim1 + 1):
                for tdim2 in range(tplotmindim2, tplotmaxdim2 + 1):
                    table[row, 0] = tdim1
                    table[row, 1] = tdim2
                    table[row, 2] = rdim1
                    table[row, 3] = rdim2
                    table[row, 4] = Wpt[tdim1, tdim2, rdim1, rdim2]
                    if deltaw[tdim1, tdim2, rdim1, rdim2] >= 0:
                        table[row, 5] = 1
                    else:
                        table[row, 5] = 0
                    row += 1
    return table


def tabulate_field_centre():
    table = np.zeros([nTdim1 * nTdim2, 4])
    row = 0
    for tdim1 in range(tmindim1, tmaxdim1 + 1):
        for tdim2 in range(tmindim2, tmaxdim2 + 1):
            table[row, 0] = tdim1
            table[row, 1] = tdim2
            table[row, 2] = Fieldcentre[0, tdim1, tdim2]
            table[row, 3] = Fieldcentre[1, tdim1, tdim2]
            row += 1
    return table


plt.subplot(2, 3, 2)
plt.title('Synaptic Weight Map')
plot = tabulate_weight_matrix()
plt.scatter(plot[:, Tplotdim - 1], plot[:, Rplotdim + 1], s=(plot[:, 4]) * 100, marker='s', c=(plot[:, 5]),
            cmap='Greys',
            edgecolors='k')
plt.clim(0, 1)
plt.ylabel('Retinal Cell Number (Dimension %d)' % (Rplotdim))
plt.xlabel('Tectal Cell Number (Dimension %d)' % (Tplotdim))
plt.xlim([tplotmin, tplotmax])
plt.ylim([rplotmin, rplotmax])

plt.subplot(2, 3, 3)
plt.title('Receptive Field Locations')
plot = tabulate_field_centre()
plt.scatter(plot[:, 2], plot[:, 3], c=(plot[:, 0] / plot[:, 1]), s=10, edgecolors='none')
plt.clim(0.1, 10)
plt.xlabel('Retinal Cell Number Dimension 1')
plt.ylabel('Retinal Cell Number Dimension 2')
plt.xlim([tmindim1, tmaxdim1])
plt.ylim([rmindim1, rmaxdim1])

plt.subplot(2, 3, 4)
plt.title('Receptive Field Separation')
plt.plot(Time, Fieldseparation)
plt.ylabel('Mean Receptive Field Separation')
plt.xlabel('Time')

plt.subplot(2, 3, 5)
plt.title('Receptive Field Size')
plt.plot(Time, Fieldsize)
plt.ylabel('Mean Receptive Field Area')
plt.xlabel('Time')

plt.subplot(2, 3, 6)
plt.title('Orientation')
plt.plot(Time, Orientation)
plt.ylabel('Oreintation')
plt.xlabel('Time')

###################### END ########################
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')

params = {'font.size': '10'}
plt.rcParams.update(params)
plt.tight_layout()
plt.show()
