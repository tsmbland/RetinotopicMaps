import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

####################### IMPORT DATA ######################

JobID = int(input('JobID: '))

Cra = np.load('../../RetinotopicMapsData/%s/EphA.npy' % ('{0:04}'.format(JobID)))
Crb = np.load('../../RetinotopicMapsData/%s/EphB.npy' % ('{0:04}'.format(JobID)))
Cta = np.load('../../RetinotopicMapsData/%s/EphrinA.npy' % ('{0:04}'.format(JobID)))
Ctb = np.load('../../RetinotopicMapsData/%s/EphrinB.npy' % ('{0:04}'.format(JobID)))

######################## PLOT OPTIONS #####################

EphAslice = (len(Cra[0, :]) - 2) // 2
EphBslice = (len(Crb[:, 0]) - 2) // 2
EphrinAslice = (len(Cta[0, 0, :]) - 2) // 2
EphrinBslice = (len(Ctb[0, :, 0]) - 2) // 2

######################### PLOT ############################
ymax1 = Cra.max()
ymax2 = Crb.max()
ymax3 = Cta[0, :, :].max()
ymax4 = Ctb[0, :, :].max()

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.scatter(range(1, len(Cra[:, 0]) - 1), Cra[1:len(Cra[:, 0]) - 1, EphAslice], c='k')
ax1.set_xlabel('Retinal Cell Number (Dimension 1)')
ax1.set_ylabel('EphA  Receptor  Density')
ax1.set_ylim(0, ymax1)
ax1.set_xlim(0, len(Cra[:, 0]) - 2)
ax1.set_title('A', x=0)

ax2 = fig.add_subplot(222)
ax2.scatter(range(1, len(Crb[0, :]) - 1), Crb[EphBslice, 1:len(Crb[0, :]) - 1], c='k')
ax2.set_xlabel('Retinal Cell Number (Dimension 2)')
ax2.set_ylabel('EphB  Receptor  Density')
ax2.set_ylim(0, ymax2)
ax2.set_xlim(0, len(Crb[0, :]) - 2)
ax2.set_title('B', x=0)

ax3 = fig.add_subplot(223)
ax3.scatter(range(1, len(Cta[0, :, 0]) - 1), Cta[0, 1:len(Cta[0, :, 0]) - 1, EphAslice], c='k')
ax3.set_xlabel('Tectal Cell Number (Dimension 1)')
ax3.set_ylabel('ephrinA  Ligand  Density')
ax3.set_ylim(0, ymax3)
ax3.set_xlim(0, len(Cta[0, :, 0]) - 2)
ax3.set_title('C', x=0)

ax4 = fig.add_subplot(224)
ax4.scatter(range(1, len(Ctb[0, 0, :]) - 1), Ctb[0, EphBslice, 1:len(Ctb[0, 0, :]) - 1], c='k')
ax4.set_xlabel('Tectal Cell Number (Dimension 2)')
ax4.set_ylabel('ephrinB  Ligand  Density')
ax4.set_ylim(0, ymax4)
ax4.set_xlim(0, len(Ctb[0, 0, :]) - 2)
ax4.set_title('D', x=0)

plt.show()
