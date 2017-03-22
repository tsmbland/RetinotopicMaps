import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns

####################### IMPORT DATA ######################

JobID = int(input('JobID: '))

Concmatrix = np.load('../../RetinotopicMapsData/%s/TectalConcentrations.npy' % ('{0:04}'.format(JobID)))

######################## PLOT OPTIONS #####################

TRin = np.load('../../RetinotopicMapsData/%s/PrimaryTR.npy' % ('{0:04}'.format(JobID)))

Tplotdim = 1
Tplotslice = 1  # (len(Concmatrix[0, 0, 0, :]) - 2) // 2

####################### PLOT ########################
ymax = Concmatrix.max()

if Tplotdim == 1:
    tplotmindim1 = tplotmin = 1
    tplotmaxdim1 = tplotmax = len(Concmatrix[0, 0, :, 0]) - 2
    tplotmindim2 = Tplotslice
    tplotmaxdim2 = Tplotslice
elif Tplotdim == 2:
    tplotmindim1 = Tplotslice
    tplotmaxdim1 = Tplotslice
    tplotmindim2 = tplotmin = 1
    tplotmaxdim2 = tplotmax = len(Concmatrix[0, 0, 0, :]) - 2

fig = plt.figure()
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.25, bottom=0.25)
axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
sframe = Slider(axframe, 'Iteration', 0, len(Concmatrix[0, :, 0, 0]) * TRin - TRin, valinit=0, valfmt='%d')


def concplot(i):
    if Tplotdim == 1:
        ax.set_xlim(1, len(Concmatrix[0, 0, :, 0]) - 2)
        for m in range(len(Concmatrix[:, 0, 0, 0])):
            ax.plot(range(tplotmin, tplotmax + 1), Concmatrix[m, i, tplotmin:tplotmax + 1, Tplotslice])

    elif Tplotdim == 2:
        ax.set_xlim(1, len(Concmatrix[0, 0, 0, :]) - 2)
        for m in range(len(Concmatrix[:, 0, 0, 0])):
            ax.plot(range(tplotmin, tplotmax + 1), Concmatrix[m, i, Tplotslice, tplotmin:tplotmax + 1])

    ax.set_xlabel('Tectal Cell Number (Dimension %d)' % (Tplotdim))
    ax.set_ylabel('Marker Concentration')
    ax.set_ylim(0, ymax)


def update(val):
    ax.clear()
    it = np.floor(sframe.val) // TRin
    concplot(it)


concplot(0)

sframe.on_changed(update)
plt.show()
