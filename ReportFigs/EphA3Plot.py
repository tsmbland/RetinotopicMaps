import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

####################### IMPORT DATA ######################

JobID = 859  # int(input('JobID: '))

print('Loading Data...')
Cra = np.load('../../RetinotopicMapsData/%s/EphA.npy' % ('{0:04}'.format(JobID)))
Crb = np.load('../../RetinotopicMapsData/%s/EphB.npy' % ('{0:04}'.format(JobID)))
EphA3 = np.load('../../RetinotopicMapsData/%s/EphA3.npy' % ('{0:04}'.format(JobID)))
Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
Fieldcentres = np.load('../../RetinotopicMapsData/%s/ReverseFieldCentres.npy' % ('{0:04}'.format(JobID)))
TRin = np.load('../../RetinotopicMapsData/%s/ReverseSecondaryTR.npy' % ('{0:04}'.format(JobID)))

fig = plt.figure()

##################### EphA3 PLOT #####################

Slicepoint = (len(Crb[:, 0]) - 2) // 2
ymax1 = Cra.max()

ax1 = fig.add_subplot(131)

ax1.scatter(range(1, len(Cra[:, 0]) - 1), Cra[1:len(Cra[:, 0]) - 1, Slicepoint],
            s=EphA3[1:len(Cra[:, 0]) * 15, Slicepoint], c='r', label='EphA3+', edgecolors='r')

ax1.scatter(range(1, len(Cra[:, 0]) - 1), Cra[1:len(Cra[:, 0]) - 1, Slicepoint],
            s=(1 - EphA3[1:len(Cra[:, 0]) * 15, Slicepoint]), c='b', label='EphA3-', edgecolors='b')

ax1.set_xlabel('Nasal - Temporal (i)')
ax1.set_ylabel('EphA Receptor Density')
ax1.set_ylim(0, ymax1)
ax1.set_xlim(0, len(Cra[:, 0]) - 2)
ax1.set_title('A', x=0)
lgnd = plt.legend(loc=2)
lgnd.legendHandles[0]._sizes = [15]
lgnd.legendHandles[1]._sizes = [15]

#################### WEIGHT PLOT ###################

# Options
Rplotdim = 1  # retina dimension plotted (1 or 2)
Rplotslice = (len(Weightmatrix[0, 0, 0, 0, :]) - 2) // 2  # slice location in the other dimension
Tplotdim = 1
Tplotslice = (len(Weightmatrix[0, 0, :, 0, 0]) - 2) // 2

if Rplotdim == 1:
    rplotmindim1 = rplotmin = 1
    rplotmaxdim1 = rplotmax = len(Weightmatrix[0, 0, 0, :, 0]) - 2
    rplotmindim2 = Rplotslice
    rplotmaxdim2 = Rplotslice
elif Rplotdim == 2:
    rplotmindim1 = Rplotslice
    rplotmaxdim1 = Rplotslice
    rplotmindim2 = rplotmin = 1
    rplotmaxdim2 = rplotmax = len(Weightmatrix[0, 0, 0, 0, :]) - 2
if Tplotdim == 1:
    tplotmindim1 = tplotmin = 1
    tplotmaxdim1 = tplotmax = len(Weightmatrix[0, :, 0, 0, 0]) - 2
    tplotmindim2 = Tplotslice
    tplotmaxdim2 = Tplotslice
elif Tplotdim == 2:
    tplotmindim1 = Tplotslice
    tplotmaxdim1 = Tplotslice
    tplotmindim2 = tplotmin = 1
    tplotmaxdim2 = tplotmax = len(Weightmatrix[0, 0, :, 0, 0]) - 2

# Tabulate Weight Matrix
table = np.zeros([len(Weightmatrix[:, 0, 0, 0, 0]), (rplotmax - rplotmin + 1) * (tplotmax - tplotmin + 1), 6])
for iteration in range(len(Weightmatrix[:, 0, 0, 0, 0])):
    row = 0
    for rdim1 in range(rplotmindim1, rplotmaxdim1 + 1):
        for rdim2 in range(rplotmindim2, rplotmaxdim2 + 1):
            for tdim1 in range(tplotmindim1, tplotmaxdim1 + 1):
                for tdim2 in range(tplotmindim2, tplotmaxdim2 + 1):
                    if Weightmatrix[iteration, tdim1, tdim2, rdim1, rdim2] != 0.:
                        table[iteration, row, 0] = tdim1
                        table[iteration, row, 1] = tdim2
                        table[iteration, row, 2] = rdim1
                        table[iteration, row, 3] = rdim2
                        table[iteration, row, 4] = Weightmatrix[iteration, tdim1, tdim2, rdim1, rdim2]
                        if EphA3[rdim1, rdim2] == 1:
                            table[iteration, row, 5] = 1
                        else:
                            table[iteration, row, 5] = 0
                        row += 1

# Plot
ax = fig.add_subplot(132)
ax.set_title('B', x=0)


def weightplot(i):
    wplot = ax.scatter(table[i // TRin, :, Rplotdim + 1], table[i // TRin, :, Tplotdim - 1],
                       s=table[i // TRin, :, 4] * 100 * table[i // TRin, :, 5], marker='s', c='r', edgecolors='r')

    wplot = ax.scatter(table[i // TRin, :, Rplotdim + 1], table[i // TRin, :, Tplotdim - 1],
                       s=table[i // TRin, :, 4] * 100 * (1 - table[i // TRin, :, 5]), marker='s', c='b',
                       edgecolors='b')

    if Rplotdim == 1:
        ax.set_xlim(1, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    elif Rplotdim == 2:
        ax.set_xlim(1, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
    if Tplotdim == 1:
        ax.set_ylim(1, len(Weightmatrix[0, :, 0, 0, 0]) - 2)
    elif Tplotdim == 2:
        ax.set_ylim(1, len(Weightmatrix[0, 0, :, 0, 0]) - 2)
    ax.set_xlabel('RGC: Nasal - Temporal (i)')
    ax.set_ylabel('SC Cell: Posterior - Anterior (m)')


weightplot(10000)

##################### FIELD PLOT ####################

ax2 = fig.add_subplot(133)
ax2.set_ylim(1, len(Weightmatrix[0, :, 0, 0, 0]) - 2)
ax2.set_xlim(1, len(Weightmatrix[0, 0, :, 0, 0]) - 2)
ax2.set_ylabel('Posterior - Anterior (m)')
ax2.set_xlabel('Lateral - Medial (n)')
ax2.set_title('C', x=0)


def fieldplot(i):
    # Normal Cells
    for rdim1 in range(len(Fieldcentres[0, i, :, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for rdim2 in range(len(Fieldcentres[0, i, 0, :])):
            if Fieldcentres[0, i, rdim1, rdim2] != 0 and Fieldcentres[1, i, rdim1, rdim2] != 0 and EphA3[
                rdim1, rdim2] == 0:
                fieldlistdim1.append(Fieldcentres[0, i, rdim1, rdim2])
                fieldlistdim2.append(Fieldcentres[1, i, rdim1, rdim2])

        ax2.plot(fieldlistdim2, fieldlistdim1, c='b', lw='0.5')

    for rdim2 in range(len(Fieldcentres[0, i, 0, :])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for rdim1 in range(len(Fieldcentres[0, i, :, 0])):
            if Fieldcentres[0, i, rdim1, rdim2] != 0 and Fieldcentres[1, i, rdim1, rdim2] != 0 and EphA3[
                rdim1, rdim2] == 0:
                fieldlistdim1.append(Fieldcentres[0, i, rdim1, rdim2])
                fieldlistdim2.append(Fieldcentres[1, i, rdim1, rdim2])

        ax2.plot(fieldlistdim2, fieldlistdim1, c='b', lw='0.5')

    # EphA3 Cells
    for rdim1 in range(len(Fieldcentres[0, i, :, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for rdim2 in range(len(Fieldcentres[0, i, 0, :])):
            if Fieldcentres[0, i, rdim1, rdim2] != 0 and Fieldcentres[1, i, rdim1, rdim2] != 0 and EphA3[
                rdim1, rdim2] == 1:
                fieldlistdim1.append(Fieldcentres[0, i, rdim1, rdim2])
                fieldlistdim2.append(Fieldcentres[1, i, rdim1, rdim2])

        ax2.plot(fieldlistdim2, fieldlistdim1, c='r', lw='0.5')

    for rdim2 in range(len(Fieldcentres[0, i, 0, :])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for rdim1 in range(len(Fieldcentres[0, i, :, 0])):
            if Fieldcentres[0, i, rdim1, rdim2] != 0 and Fieldcentres[1, i, rdim1, rdim2] != 0 and EphA3[
                rdim1, rdim2] == 1:
                fieldlistdim1.append(Fieldcentres[0, i, rdim1, rdim2])
                fieldlistdim2.append(Fieldcentres[1, i, rdim1, rdim2])

        ax2.plot(fieldlistdim2, fieldlistdim1, c='r', lw='0.5')


fieldplot(len(Fieldcentres[0, :, 0, 0]) - 1)

plt.show()
