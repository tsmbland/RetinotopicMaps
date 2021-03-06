import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

start = time.time()

##################### IMPORT DATA ########################

JobID = 813
Tdim1 = 25
Tdim2 = 25
print('Loading Data...')
Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
Fieldcentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(JobID)))
Fieldsizes = np.load('../../RetinotopicMapsData/%s/FieldSizes.npy' % ('{0:04}'.format(JobID)))
TRin1 = np.load('../../RetinotopicMapsData/%s/PrimaryTR.npy' % ('{0:04}'.format(JobID)))
TRin2 = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

######################## PLOT OPTIONS #####################

Iterations = [20, 40, 60, 500]

######################## TABLE #########################

# Tabulate Weight Matrix
table = np.zeros(
    [len(Weightmatrix[:, 0, 0, 0, 0]), len(Weightmatrix[0, 0, 0, :, 0]) * len(Weightmatrix[0, 0, 0, 0, :]), 4])

for iteration in range(len(Weightmatrix[:, 0, 0, 0, 0]) - 1):
    row = 0
    deltaw = Weightmatrix[iteration + 1, :, :, :, :] - Weightmatrix[iteration, :, :, :, :]
    for rdim1 in range(1, len(Weightmatrix[0, 0, 0, :, 0]) - 1):
        for rdim2 in range(1, len(Weightmatrix[0, 0, 0, 0, :]) - 1):
            if Weightmatrix[iteration, Tdim1, Tdim2, rdim1, rdim2] != 0.:
                table[iteration, row, 0] = rdim1
                table[iteration, row, 1] = rdim2
                table[iteration, row, 2] = Weightmatrix[iteration, Tdim1, Tdim2, rdim1, rdim2]
                row += 1

    sys.stdout.write('\rProcessing data... %i percent' % (iteration * 100 / len(Weightmatrix[:, 0, 0, 0, 0])))
    sys.stdout.flush()

####################### PLOT ##########################
fig = plt.figure()
figs = fig.add_subplot(111, frameon=False)
figs.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
figs.set_ylabel('Dorsal - Ventral (j)')
figs.set_xlabel('Nasal - Temporal (i)')


def areaplot(plotn):
    i = Iterations[plotn]

    ax = fig.add_subplot(2, 2, plotn + 1)
    ax.set_xlim(0, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    ax.set_ylim(0, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
    ax.set_title('%d iterations' % i)

    wplot = ax.scatter(table[i // TRin1, :, 0], table[i // TRin1, :, 1], s=(table[i // TRin1, :, 2]) * 100, marker='s',
                       c='k', cmap='Greys')

    origin = ax.scatter(Fieldcentres[0, i // TRin2, Tdim1, Tdim2], Fieldcentres[1, i // TRin2, Tdim1, Tdim2], c='r')
    circle = plt.Circle((Fieldcentres[0, i // TRin2, Tdim1, Tdim2], Fieldcentres[1, i // TRin2, Tdim1, Tdim2]),
                        Fieldsizes[i // TRin2, Tdim1, Tdim2] / 2,
                        fill=False, color='r')

    ax.add_artist(circle)

    if (plotn + 1) == 1:
        ax.set_xticklabels([])
    if (plotn + 1) == 2:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if (plotn + 1) == 4:
        ax.set_yticklabels([])


for plotn in range(len(Iterations)):
    areaplot(plotn)

plt.show()
