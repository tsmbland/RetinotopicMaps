import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

##################### IMPORT DATA #####################

JobID = 817
EB = 'y'  # input('Exclude Boundaries? (y/n) ')

if EB == 'n':
    Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSize.npy' % ('{0:04}'.format(JobID)))
    Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparation.npy' % ('{0:04}'.format(JobID)))
    FieldseparationStdev = np.load('../../RetinotopicMapsData/%s/FieldSeparationStdev.npy' % ('{0:04}'.format(JobID)))
    Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatch.npy' % ('{0:04}'.format(JobID)))

    Fieldsizechange = np.load('../../RetinotopicMapsData/%s/FieldSizeChange.npy' % ('{0:04}'.format(JobID)))
    Fieldseparationchange = np.load('../../RetinotopicMapsData/%s/FieldSeparationChange.npy' % ('{0:04}'.format(JobID)))
    Systemsmatchchange = np.load('../../RetinotopicMapsData/%s/SystemsMatchChange.npy' % ('{0:04}'.format(JobID)))

elif EB == 'y':
    Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(JobID)))
    Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(JobID)))
    FieldseparationStdev = np.load('../../RetinotopicMapsData/%s/FieldSeparationStdevEB.npy' % ('{0:04}'.format(JobID)))
    Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(JobID)))

    Fieldsizechange = np.load('../../RetinotopicMapsData/%s/FieldSizeChangeEB.npy' % ('{0:04}'.format(JobID)))
    Fieldseparationchange = np.load(
        '../../RetinotopicMapsData/%s/FieldSeparationChangeEB.npy' % ('{0:04}'.format(JobID)))
    Systemsmatchchange = np.load(
        '../../RetinotopicMapsData/%s/SystemsMatchChangeEB.npy' % ('{0:04}'.format(JobID)))

TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

######################## PLOTS #######################
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.set_title('A', x=0)
ax1.plot(range(TRin, len(Fieldseparation) * TRin, TRin), Fieldseparation[1:], c='k')
ax1.set_ylabel('Mean Receptive Field Separation')
ax1.set_xlabel('Iterations')
ax1.set_xlim(0, 5000)
ax1.set_ylim(0, 3)

ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2.set_title('B', x=0)
ax2.plot(range(TRin, len(Fieldsize) * TRin, TRin), Fieldsize[1:], c='k')
ax2.set_ylabel('Mean Receptive Field Diameter')
ax2.set_xlabel('Iterations')
ax2.set_xlim(0, 5000)

ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3.set_title('C', x=0)
ax3.plot(range(TRin, len(Systemsmatch) * TRin, TRin), Systemsmatch[1:], c='k')
ax3.set_ylabel('Systems Match Score')
ax3.set_xlabel('Iterations')
ax3.set_xlim(0, 2000)
ax3.set_ylim(0, 5)

plt.show()
