import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns

####################### IMPORT DATA ######################

JobID = int(input('JobID: '))

Cta = np.load('../../RetinotopicMapsData/%s/EphrinA.npy' % ('{0:04}'.format(JobID)))
Ctb = np.load('../../RetinotopicMapsData/%s/EphrinB.npy' % ('{0:04}'.format(JobID)))

######################## PLOT OPTIONS #####################

TRin = np.load('../../RetinotopicMapsData/%s/PrimaryTR.npy' % ('{0:04}'.format(JobID)))

EphAslice = (len(Cta[0, 0, :]) - 2) // 2
EphBslice = (len(Ctb[0, :, 0]) - 2) // 2

######################### PLOT ############################
ymax = max(Cta.max(), Ctb.max())

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plt.subplots_adjust(left=0.25, bottom=0.25)
axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
sframe = Slider(axframe, 'Iteration', 0, len(Cta[:, 0, 0])*TRin - TRin, valinit=0, valfmt='%d')


def ephrinplot(i):
    ax1.plot(range(1, len(Cta[0, :, 0]) - 1), Cta[i, 1:len(Cta[0, :, 0]) - 1, EphAslice])
    ax2.plot(range(1, len(Ctb[0, 0, :]) - 1), Ctb[i, EphBslice, 1:len(Ctb[0, 0, :]) - 1])
    ax1.set_xlabel('Tectal Cell Number (Dimension 1)')
    ax1.set_ylabel('EphrinA Concentration')
    ax2.set_xlabel('Tectal Cell Number (Dimension 2)')
    ax2.set_ylabel('EphrinB Concentration')
    ax1.set_ylim(0,ymax)
    ax2.set_ylim(0,ymax)

def update(val):
    ax1.clear()
    ax2.clear()
    it = np.floor(sframe.val)//TRin
    ephrinplot(it)

ephrinplot(0)

sframe.on_changed(update)
plt.show()