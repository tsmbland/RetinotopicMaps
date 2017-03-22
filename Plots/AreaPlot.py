import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
import seaborn as sns

####################### IMPORT DATA ######################

JobID = int(input('JobID: '))
Tdim1 = int(input('Tectal Cell (Dimension 1): '))
Tdim2 = int(input('Tectal Cell (Dimension 2): '))
print('Loading Data...')
Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
Fieldcentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(JobID)))
Fieldsizes = np.load('../../RetinotopicMapsData/%s/FieldSizes.npy' % ('{0:04}'.format(JobID)))
TRin1 = np.load('../../RetinotopicMapsData/%s/PrimaryTR.npy' % ('{0:04}'.format(JobID)))
TRin2 = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

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
                if deltaw[Tdim1, Tdim2, rdim1, rdim2] >= 0:
                    table[iteration, row, 3] = 1
                else:
                    table[iteration, row, 3] = 0
                row += 1

    sys.stdout.write('\rProcessing data... %i percent' % (iteration * 100 / len(Weightmatrix[:, 0, 0, 0, 0])))
    sys.stdout.flush()

########################## PLOT ##########################

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel('Retinal Cell Number (Dimension 2)')
ax.set_xlabel('Retinal Cell Number (Dimension 1)')
ax.set_xlim(1, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
ax.set_ylim(1, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
plt.subplots_adjust(left=0.25, bottom=0.25)
axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
sframe = Slider(axframe, 'Iteration', 0, (len(Weightmatrix[:, 0, 0, 0, 0])) * TRin1 - TRin1 - 1, valinit=0, valfmt='%d')


def weightplot(i):
    wplot = ax.scatter(table[i // TRin1, :, 0], table[i // TRin1, :, 1], s=(table[i // TRin1, :, 2]) * 50, marker='s',
                       c=(table[i // TRin1, :, 3]),
                       cmap='Greys', edgecolors='k')
    wplot.set_clim(0, 1)

    ax.set_ylim(1, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
    ax.set_xlim(1, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    ax.set_ylabel('Retinal Cell Number (Dimension 2)')
    ax.set_xlabel('Retinal Cell Number (Dimension 1)')

    circle = plt.Circle((Fieldcentres[0, i // TRin2, Tdim1, Tdim2], Fieldcentres[1, i // TRin2, Tdim1, Tdim2]),
                        Fieldsizes[i // TRin2, Tdim1, Tdim2] / 2,
                        fill=False)
    ax.add_artist(circle)


def update(val):
    ax.clear()
    i = np.floor(sframe.val)
    weightplot(i)


weightplot(0)

####################### END ########################
print('Complete!')
sys.stdout.flush()
sframe.on_changed(update)
plt.show()
