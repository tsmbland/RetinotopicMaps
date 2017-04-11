import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

####################### IMPORT DATA ######################

JobID = int(input('JobID: '))

print('Loading Data...')
Cra = np.load('../../RetinotopicMapsData/%s/EphA.npy' % ('{0:04}'.format(JobID)))
Crb = np.load('../../RetinotopicMapsData/%s/EphB.npy' % ('{0:04}'.format(JobID)))
EphA3 = np.load('../../RetinotopicMapsData/%s/EphA3.npy' % ('{0:04}'.format(JobID)))
Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
Fieldcentres = np.load('../../RetinotopicMapsData/%s/ReverseFieldCentres.npy' % ('{0:04}'.format(JobID)))
TRin = np.load('../../RetinotopicMapsData/%s/ReverseSecondaryTR.npy' % ('{0:04}'.format(JobID)))

fig = plt.figure()

##################### EphA3 PLOT #####################

Slicepoint = (len(Crb[:, 0]) - 2) // 2
ymax1 = Cra.max()

ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax1.scatter(range(1, len(Cra[:, 0]) - 1), Cra[1:len(Cra[:, 0]) - 1, Slicepoint],
            c=EphA3[1:len(Cra[:, 0]) - 1, Slicepoint], cmap='bwr')
ax1.set_xlabel('Nasal - Temporal (i)')
ax1.set_ylabel('EphA Receptor Density')
ax1.set_ylim(0, ymax1)
ax1.set_xlim(0, len(Cra[:, 0]) - 2)
ax1.set_title('A', x=0)

##################### FIELD PLOT ####################

ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=2)
ax2.set_xlim(1, len(Weightmatrix[0, :, 0, 0, 0]) - 2)
ax2.set_ylim(1, len(Weightmatrix[0, 0, :, 0, 0]) - 2)
ax2.set_xlabel('Posterior - Anterior (m)')
ax2.set_ylabel('Lateral - Medial (n)')
ax2.set_title('B', x=0)


def fieldplot(i):
    # Normal Cells
    for rdim1 in range(len(Fieldcentres[0, i, :, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for rdim2 in range(len(Fieldcentres[0, i, 0, :])):
            if Fieldcentres[0, i, rdim1, rdim2] != 0 and Fieldcentres[1, i, rdim1, rdim2] != 0 and EphA3[
                rdim1, rdim2] == 0:
                fieldlistdim1.append(Fieldcentres[0, i, rdim1, rdim2])
                fieldlistdim2.append(Fieldcentres[1, i, rdim1, rdim2])

        ax2.plot(fieldlistdim1, fieldlistdim2, c='b', lw='0.5')

    for rdim2 in range(len(Fieldcentres[0, i, 0, :])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for rdim1 in range(len(Fieldcentres[0, i, :, 0])):
            if Fieldcentres[0, i, rdim1, rdim2] != 0 and Fieldcentres[1, i, rdim1, rdim2] != 0 and EphA3[
                rdim1, rdim2] == 0:
                fieldlistdim1.append(Fieldcentres[0, i, rdim1, rdim2])
                fieldlistdim2.append(Fieldcentres[1, i, rdim1, rdim2])

        ax2.plot(fieldlistdim1, fieldlistdim2, c='b', lw='0.5')

    # EphA3 Cells
    for rdim1 in range(len(Fieldcentres[0, i, :, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for rdim2 in range(len(Fieldcentres[0, i, 0, :])):
            if Fieldcentres[0, i, rdim1, rdim2] != 0 and Fieldcentres[1, i, rdim1, rdim2] != 0 and EphA3[
                rdim1, rdim2] == 1:
                fieldlistdim1.append(Fieldcentres[0, i, rdim1, rdim2])
                fieldlistdim2.append(Fieldcentres[1, i, rdim1, rdim2])

        ax2.plot(fieldlistdim1, fieldlistdim2, c='r', lw='0.5')

    for rdim2 in range(len(Fieldcentres[0, i, 0, :])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for rdim1 in range(len(Fieldcentres[0, i, :, 0])):
            if Fieldcentres[0, i, rdim1, rdim2] != 0 and Fieldcentres[1, i, rdim1, rdim2] != 0 and EphA3[
                rdim1, rdim2] == 1:
                fieldlistdim1.append(Fieldcentres[0, i, rdim1, rdim2])
                fieldlistdim2.append(Fieldcentres[1, i, rdim1, rdim2])

        ax2.plot(fieldlistdim1, fieldlistdim2, c='r', lw='0.5')


fieldplot(len(Fieldcentres[0, :, 0, 0]) - 1)

plt.show()
