import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

####################### IMPORT DATA ######################

JobID = int(input('JobID: '))
print('Loading Data...')
Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))

fig = plt.figure()


def weightplot(Rplotdim, Rplotslice, Tplotdim, Tplotslice, plotn):
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
    table = np.zeros([(rplotmax - rplotmin + 1) * (tplotmax - tplotmin + 1), 5])
    row = 0
    for rdim1 in range(rplotmindim1, rplotmaxdim1 + 1):
        for rdim2 in range(rplotmindim2, rplotmaxdim2 + 1):
            for tdim1 in range(tplotmindim1, tplotmaxdim1 + 1):
                for tdim2 in range(tplotmindim2, tplotmaxdim2 + 1):
                    if Weightmatrix[-1, tdim1, tdim2, rdim1, rdim2] != 0.:
                        table[row, 0] = tdim1
                        table[row, 1] = tdim2
                        table[row, 2] = rdim1
                        table[row, 3] = rdim2
                        table[row, 4] = Weightmatrix[-1, tdim1, tdim2, rdim1, rdim2]
                        row += 1

        sys.stdout.flush()

    ####################### PLOT ##########################
    ax = fig.add_subplot(2, 2, plotn)
    ax.set_xlim(1, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    ax.set_ylim(1, len(Weightmatrix[0, 0, 0, 0, :]) - 2)

    wplot = ax.scatter(table[:, Tplotdim - 1], table[:, Rplotdim + 1],
                       s=(table[:, 4]) * 50, marker='s',
                       c='k', edgecolors='k')
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


weightplot(1, (len(Weightmatrix[0, 0, 0, 0, :]) - 2) // 2, 1, (len(Weightmatrix[0, 0, :, 0, 0]) - 2) // 2, 1)
weightplot(1, (len(Weightmatrix[0, 0, 0, 0, :]) - 2) // 2, 2, (len(Weightmatrix[0, :, 0, 0, 0]) - 2) // 2, 2)
weightplot(2, (len(Weightmatrix[0, 0, 0, :, 0]) - 2) // 2, 1, (len(Weightmatrix[0, 0, :, 0, 0]) - 2) // 2, 3)
weightplot(2, (len(Weightmatrix[0, 0, 0, :, 0]) - 2) // 2, 2, (len(Weightmatrix[0, :, 0, 0, 0]) - 2) // 2, 4)

plt.show()
