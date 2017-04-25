import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

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

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.scatter(range(1, len(Cra[:, 0]) - 1), Cra[1:len(Cra[:, 0]) - 1, EphAslice],
            c=EphA3[1:len(Cra[:, 0]) - 1, EphAslice], cmap='bwr')
ax1.set_xlabel('Retinal Cell Number (Dimension 1)')
ax1.set_ylabel('EphA Receptor Density')
ax1.set_ylim(0, ymax1)
ax1.set_xlim(0, len(Cra[:, 0]) - 2)

plt.show()
