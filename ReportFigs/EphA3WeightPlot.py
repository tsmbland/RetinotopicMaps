import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
import seaborn as sns

####################### IMPORT DATA ######################

JobID = 859
print('Loading Data...')
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
for iteration in range(len(Weightmatrix[:, 0, 0, 0, 0]) + 1):
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
                        row += 1

    sys.stdout.write('\rProcessing data... %i percent' % (iteration * 100 / len(Weightmatrix[:, 0, 0, 0, 0])))
    sys.stdout.flush()

########################## PLOT ##########################

fig = plt.figure()
ax = fig.add_subplot(111)


def weightplot(i):
    wplot = ax.scatter(table[i // TRin, :, Rplotdim + 1], table[i // TRin, :, Tplotdim - 1],
                       s=(table[i // TRin, :, 4]) * 100, marker='s', c='k')

    if Rplotdim == 1:
        ax.set_xlim(1, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    elif Rplotdim == 2:
        ax.set_xlim(1, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
    if Tplotdim == 1:
        ax.set_ylim(1, len(Weightmatrix[0, :, 0, 0, 0]) - 2)
    elif Tplotdim == 2:
        ax.set_ylim(1, len(Weightmatrix[0, 0, :, 0, 0]) - 2)
    ax.set_xlabel('RGC: Nasal - Temporal (i)')
    ax.set_ylabel('Tectal Cell: Posterior - Anterior (m)')


weightplot(10000)

####################### END ########################
print('Complete!')
plt.show()
