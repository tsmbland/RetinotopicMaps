import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
import seaborn as sns

start = time.time()

##################### IMPORT DATA ########################

JobID = int(input('JobID: '))

print('Loading Data...')
Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
Fieldcentres = np.load('../../RetinotopicMapsData/%s/ReverseFieldCentres.npy' % ('{0:04}'.format(JobID)))
EphA3 = np.load('../../RetinotopicMapsData/%s/EphA3.npy' % ('{0:04}'.format(JobID)))
TRin = np.load('../../RetinotopicMapsData/%s/ReverseSecondaryTR.npy' % ('{0:04}'.format(JobID)))

####################### PLOT ##########################

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(1, len(Weightmatrix[0, :, 0, 0, 0]) - 2)
ax.set_ylim(1, len(Weightmatrix[0, 0, :, 0, 0]) - 2)
ax.set_xlabel('Tectal Cell Number (Dimension 1)')
ax.set_ylabel('Tectal Cell Number (Dimension 2)')
plt.subplots_adjust(left=0.25, bottom=0.25)
axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
sframe = Slider(axframe, 'Iteration', 0, len(Fieldcentres[0, :, 0, 0]) * TRin - TRin, valinit=0, valfmt='%d')


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

        ax.plot(fieldlistdim1, fieldlistdim2, c='b')

    for rdim2 in range(len(Fieldcentres[0, i, 0, :])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for rdim1 in range(len(Fieldcentres[0, i, :, 0])):
            if Fieldcentres[0, i, rdim1, rdim2] != 0 and Fieldcentres[1, i, rdim1, rdim2] != 0 and EphA3[
                rdim1, rdim2] == 0:
                fieldlistdim1.append(Fieldcentres[0, i, rdim1, rdim2])
                fieldlistdim2.append(Fieldcentres[1, i, rdim1, rdim2])

        ax.plot(fieldlistdim1, fieldlistdim2, c='b')

    # EphA3 Cells
    for rdim1 in range(len(Fieldcentres[0, i, :, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for rdim2 in range(len(Fieldcentres[0, i, 0, :])):
            if Fieldcentres[0, i, rdim1, rdim2] != 0 and Fieldcentres[1, i, rdim1, rdim2] != 0 and EphA3[
                rdim1, rdim2] == 1:
                fieldlistdim1.append(Fieldcentres[0, i, rdim1, rdim2])
                fieldlistdim2.append(Fieldcentres[1, i, rdim1, rdim2])

        ax.plot(fieldlistdim1, fieldlistdim2, c='r')

    for rdim2 in range(len(Fieldcentres[0, i, 0, :])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for rdim1 in range(len(Fieldcentres[0, i, :, 0])):
            if Fieldcentres[0, i, rdim1, rdim2] != 0 and Fieldcentres[1, i, rdim1, rdim2] != 0 and EphA3[
                rdim1, rdim2] == 1:
                fieldlistdim1.append(Fieldcentres[0, i, rdim1, rdim2])
                fieldlistdim2.append(Fieldcentres[1, i, rdim1, rdim2])

        ax.plot(fieldlistdim1, fieldlistdim2, c='r')


def update(val):
    it = np.floor(sframe.val) // TRin
    ax.clear()
    fieldplot(it)
    ax.set_xlim(1, len(Weightmatrix[0, :, 0, 0, 0]) - 2)
    ax.set_ylim(1, len(Weightmatrix[0, 0, :, 0, 0]) - 2)


fieldplot(0)

sframe.on_changed(update)
plt.show()
