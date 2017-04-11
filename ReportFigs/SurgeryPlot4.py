import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

JobID = np.zeros([4], dtype=np.int32)
JobID[0] = int(input('Surgery 1 JobID: '))
JobID[1] = int(input('Surgery 2 JobID: '))
JobID[2] = int(input('Surgery 3 JobID: '))


fig = plt.figure()


#################### FIELD PLOTS ######################

def fieldplot(JobID, plotn):
    Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
    Fieldcentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(JobID)))

    ax = fig.add_subplot(2, 3, plotn + 1)
    ax.set_xlim(0, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    ax.set_ylim(0, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
    ax.set_ylabel('Retinal Cell Number (Dimension 2)')
    ax.set_xlabel('Retinal Cell Number (Dimension 1)')
    Titles = ['A', 'B', 'C']
    ax.set_title(Titles[plotn])

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


for plotn in range(3):
    fieldplot(JobID[plotn], plotn)

################## PRECISION PLOTS ##################

# Import data
Fieldseparation1 = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(JobID[0])))
Fieldsize1 = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(JobID[0])))
Systemsmatch1 = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(JobID[0])))
TRin1 = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID[0])))

Fieldseparation2 = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(JobID[1])))
Fieldsize2 = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(JobID[1])))
Systemsmatch2 = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(JobID[1])))
TRin2 = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID[1])))

Fieldseparation3 = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(JobID[2])))
Fieldsize3 = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(JobID[2])))
Systemsmatch3 = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(JobID[2])))
TRin3 = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID[2])))

# Separation Plot
ax = fig.add_subplot(2, 3, 4)
ax.plot(range(TRin1, len(Fieldseparation1) * TRin1, TRin1), Fieldseparation1[1:],
        range(TRin2, len(Fieldseparation2) * TRin2, TRin2), Fieldseparation2[1:],
        range(TRin3, len(Fieldseparation3) * TRin3, TRin3), Fieldseparation3[1:])
ax.set_ylabel('Mean Receptive Field Separation')
ax.set_xlabel('Iterations')
ax.set_title('D')

# Size Plot
ax = fig.add_subplot(2, 3, 5)
ax.plot(range(TRin1, len(Fieldsize1) * TRin1, TRin1), Fieldsize1[1:],
        range(TRin2, len(Fieldsize2) * TRin2, TRin2), Fieldsize2[1:],
        range(TRin3, len(Fieldsize3) * TRin3, TRin3), Fieldsize3[1:])
ax.set_ylabel('Mean Receptive Field Size')
ax.set_xlabel('Iterations')
ax.set_title('E')

# Systems Match Plot
ax = fig.add_subplot(2, 3, 6)
ax.plot(range(TRin1, len(Systemsmatch1) * TRin1, TRin1), Systemsmatch1[1:],
        range(TRin2, len(Systemsmatch2) * TRin2, TRin2), Systemsmatch2[1:],
        range(TRin3, len(Systemsmatch3) * TRin3, TRin3), Systemsmatch3[1:])
ax.set_ylabel('Systems Match Score')
ax.set_xlabel('Iterations')
ax.set_title('F')

plt.show()
