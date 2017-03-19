import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys

####################### IMPORT DATA ######################

JobID = int(input('JobID: '))
Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))

######################## PLOT OPTIONS #####################

TRin = np.load('../../RetinotopicMapsData/%s/PrimaryTR.npy' % ('{0:04}'.format(JobID)))

Rplotdim = 1  # retina dimension plotted (1 or 2)
Rplotslice = (len(Weightmatrix[0, 0, 0, 0, :]) - 2) // 2  # slice location in the other dimension
Tplotdim = 1
Tplotslice = (len(Weightmatrix[0, 0, :, 0, 0]) - 2) // 2

######################## TABLE #########################
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
for iteration in range(len(Weightmatrix[:, 0, 0, 0, 0]) - 1):
    row = 0
    deltaw = Weightmatrix[iteration + 1, :, :, :, :] - Weightmatrix[iteration, :, :, :, :]
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
                        if deltaw[tdim1, tdim2, rdim1, rdim2] >= 0:
                            table[iteration, row, 5] = 1
                        else:
                            table[iteration, row, 5] = 0
                        row += 1

    sys.stdout.write('\rProcessing data... %i percent' % (iteration * 100 / len(Weightmatrix[:, 0, 0, 0, 0])))
    sys.stdout.flush()

########################## PLOT ##########################

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel('Retinal Cell Number (Dimension %d)' % (Rplotdim))
ax.set_xlabel('Tectal Cell Number (Dimension %d)' % (Tplotdim))
ax.set_xlim(1, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
ax.set_ylim(1, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
plt.subplots_adjust(left=0.25, bottom=0.25)
axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
sframe = Slider(axframe, 'Iteration', 0, (len(Weightmatrix[:, 0, 0, 0, 0])) * TRin - TRin - 1, valinit=0, valfmt='%d')


def weightplot(i):
    wplot = ax.scatter(table[i, :, Tplotdim - 1], table[i, :, Rplotdim + 1], s=(table[i, :, 4]) * 50, marker='s',
                       c=(table[i, :, 5]),
                       cmap='Greys', edgecolors='k')
    wplot.set_clim(0, 1)
    if Rplotdim == 1:
        ax.set_ylim(1, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    elif Rplotdim == 2:
        ax.set_ylim(1, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
    if Tplotdim == 1:
        ax.set_xlim(1, len(Weightmatrix[0, :, 0, 0, 0]) - 2)
    elif Tplotdim == 2:
        ax.set_xlim(1, len(Weightmatrix[0, 0, :, 0, 0]) - 2)
    ax.set_ylabel('Retinal Cell Number (Dimension %d)' % (Rplotdim))
    ax.set_xlabel('Tectal Cell Number (Dimension %d)' % (Tplotdim))


def update(val):
    ax.clear()
    it = np.floor(sframe.val) // TRin
    weightplot(it)


weightplot(0)

####################### END ########################
sys.stdout.write('\rComplete!')
sys.stdout.flush()
sframe.on_changed(update)
plt.show()
