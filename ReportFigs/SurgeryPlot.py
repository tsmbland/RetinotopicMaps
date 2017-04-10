import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

JobID = np.zeros([4], dtype=np.int32)
JobID[0] = int(input('Surgery 1 JobID: '))
JobID[1] = int(input('Surgery 2 JobID: '))
JobID[2] = int(input('Surgery 3 JobID: '))
JobID[3] = int(input('Surgery 4 JobID: '))

####################### PLOT ##########################
fig = plt.figure()
figs = fig.add_subplot(111, frameon=False)
figs.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
figs.set_ylabel('Retinal Cell Number (Dimension 2)')
figs.set_xlabel('Retinal Cell Number (Dimension 1)')


def fieldplot(JobID, plotn):
    Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
    Fieldcentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(JobID)))

    ax = fig.add_subplot(2, 2, plotn + 1)
    ax.set_xlim(0, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    ax.set_ylim(0, len(Weightmatrix[0, 0, 0, 0, :]) - 2)

    for tdim1 in range(len(Fieldcentres[0, -1, :, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim2 in range(len(Fieldcentres[0, -1, 0, :])):
            if Fieldcentres[0, -1, tdim1, tdim2] != 0 and Fieldcentres[1, -1, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, -1, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, -1, tdim1, tdim2])
        ax.plot(fieldlistdim1, fieldlistdim2, c='k', lw='0.5')

    for tdim2 in range(len(Fieldcentres[0, -1, 0, :])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim1 in range(len(Fieldcentres[0, -1, :, 0])):
            if Fieldcentres[0, -1, tdim1, tdim2] != 0 and Fieldcentres[1, -1, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, -1, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, -1, tdim1, tdim2])
        ax.plot(fieldlistdim1, fieldlistdim2, c='k', lw='0.5')

    if (plotn + 1) == 1:
        ax.set_xticklabels([])
        ax.set_title('A', x=0)
    if (plotn + 1) == 2:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title('B', x=0)
    if (plotn + 1) == 3:
        ax.set_title('C', x=0)
    if (plotn + 1) == 4:
        ax.set_yticklabels([])
        ax.set_title('D', x=0)


for plotn in range(4):
    fieldplot(JobID[plotn], plotn)

plt.show()
