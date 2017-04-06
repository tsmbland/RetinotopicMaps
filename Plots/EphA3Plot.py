import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns

####################### IMPORT DATA ######################

JobID = int(input('JobID: '))

Cra = np.load('../../RetinotopicMapsData/%s/EphA.npy' % ('{0:04}'.format(JobID)))
Crb = np.load('../../RetinotopicMapsData/%s/EphB.npy' % ('{0:04}'.format(JobID)))
EphA3 = np.load('../../RetinotopicMapsData/%s/EphA3.npy' % ('{0:04}'.format(JobID)))

######################## PLOT OPTIONS #####################

EphAslice = (len(Cra[0, :]) - 2) // 2
EphBslice = (len(Crb[:, 0]) - 2) // 2

######################### PLOT ############################
ymax1 = Cra.max()
ymax2 = Crb.max()

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(range(1, len(Cra[:, 0]) - 1), Cra[1:len(Cra[:, 0]) - 1, EphAslice], c=EphA3[1:len(Cra[:, 0]) - 1, EphAslice], cmap='bwr')
ax2.scatter(range(1, len(Crb[0, :]) - 1), Crb[EphBslice, 1:len(Crb[0, :]) - 1])
ax1.set_xlabel('Tectal Cell Number (Dimension 1)')
ax1.set_ylabel('EphrinA Concentration')
ax2.set_xlabel('Tectal Cell Number (Dimension 2)')
ax2.set_ylabel('EphrinB Concentration')
ax1.set_ylim(0, ymax1)
ax2.set_ylim(0, ymax2)
ax1.set_xlim(0, len(Cra[:, 0]) - 2)
ax2.set_xlim(0, len(Crb[0, :]) - 2)

plt.show()