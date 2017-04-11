import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

JobID = np.zeros([4], dtype=np.int32)
JobID[0] = int(input('Surgery 1 JobID: '))
JobID[1] = int(input('Surgery 2 JobID: '))
JobID[2] = int(input('Surgery 3 JobID: '))

Titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']


####################### PLOT ##########################


# Field Plot

def fieldplot(JobID, plotn):
    Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
    Fieldcentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(JobID)))

    ax = plt.subplot2grid((9, 6), (3 * plotn, 0), colspan=3, rowspan=3)
    ax.set_xlim(0, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    ax.set_ylim(0, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
    ax.set_ylabel('Retinal Cell Number (Dimension 2)')
    ax.set_xlabel('Retinal Cell Number (Dimension 1)')
    ax.set_title(Titles[plotn + 1])

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


# Precision Plots

def separationplot(JobID, plotn):
    Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(JobID)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

    ax = plt.subplot2grid((9, 6), (3 * plotn, 3), colspan=3)
    ax.plot(range(TRin, len(Fieldseparation) * TRin, TRin), Fieldseparation[1:], c='k')
    ax.set_ylabel('Mean Receptive Field Separation')
    ax.set_xlabel('Iterations')
    ax.set_title(Titles[plotn + 2])


def sizeplot(JobID, plotn):
    Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(JobID)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

    ax = plt.subplot2grid((9, 6), (3 * plotn + 1, 3), colspan=3)
    ax.plot(range(TRin, len(Fieldsize) * TRin, TRin), Fieldsize[1:], c='k')
    ax.set_ylabel('Mean Receptive Field Size')
    ax.set_xlabel('Iterations')
    ax.set_title(Titles[plotn + 3])


def systemsmatchplot(JobID, plotn):
    Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(JobID)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

    ax = plt.subplot2grid((9, 6), (3 * plotn + 1, 3), colspan=3)
    ax.plot(range(TRin, len(Systemsmatch) * TRin, TRin), Systemsmatch[1:], c='k')
    ax.set_ylabel('Systems Match Score')
    ax.set_xlabel('Iterations')
    ax.set_title(Titles[plotn + 4])


for plotn in range(3):
    fieldplot(JobID[plotn], plotn)
    separationplot(JobID[plotn], plotn)
    sizeplot(JobID[plotn], plotn)
    systemsmatchplot(JobID[plotn], plotn)

plt.show()
