import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

#################### JOBS ###################

yLT = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

JobID = np.zeros([11, 3], dtype=np.int32)
JobID[0] = [873, 884, 895]
JobID[1] = [872, 883, 894]
JobID[2] = [871, 882, 893]
JobID[3] = [870, 881, 892]
JobID[4] = [869, 880, 891]
JobID[5] = [868, 879, 890]
JobID[6] = [867, 878, 889]
JobID[7] = [866, 877, 888]
JobID[8] = [865, 876, 887]
JobID[9] = [864, 875, 886]
JobID[10] = [863, 874, 885]

################# DATA MATRICES ###############
SepEB = np.zeros([11, 3])
SizeEB = np.zeros([11, 3])
SMEB = np.zeros([11, 3])

################### IMPORT DATA ################

for column in range(11):
    for row in range(3):
        ID = JobID[column, row]

        # Import Data
        FieldsizeEB = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(ID)))
        FieldseparationEB = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(ID)))
        SystemsmatchEB = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(ID)))

        SepEB[column, row] = FieldseparationEB[-1]
        SizeEB[column, row] = FieldsizeEB[-1]
        SMEB[column, row] = SystemsmatchEB[-1]

################## ORDER DATA ################

for column in range(11):
    SepEB[column, :] = np.sort(SepEB[column, :])
    SizeEB[column, :] = np.sort(SizeEB[column, :])
    SMEB[column, :] = np.sort(SMEB[column, :])

##################### PLOT ###################
fig = plt.figure()

# Field Separation Plot
ax1 = fig.add_subplot(131)

min = SepEB[:, 0]
med = SepEB[:, 1]
max = SepEB[:, 2]

ax1.plot(yLT, min, c='0.8')
ax1.plot(yLT, med, c='k')
ax1.plot(yLT, max, c='0.8')

ax1.fill_between(yLT, min, med, color='0.8')
ax1.fill_between(yLT, med, max, color='0.8')

ax1.set_xlabel('Scaling Factor')
ax1.set_ylabel('Mean Receptive Field Separation')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 3.5)
ax1.set_title('A', x=0)

# Field Size Plot
ax2 = fig.add_subplot(132)

min = SizeEB[:, 0]
med = SizeEB[:, 1]
max = SizeEB[:, 2]

ax2.plot(yLT, min, c='0.8')
ax2.plot(yLT, med, c='k')
ax2.plot(yLT, max, c='0.8')

ax2.fill_between(yLT, min, med, color='0.8')
ax2.fill_between(yLT, med, max, color='0.8')

ax2.set_xlabel('Scaling Factor')
ax2.set_ylabel('Mean Receptive Field Size')
ax2.set_xlim(0, 1)
ax2.set_title('B', x=0)

# Systems Match Plot
ax3 = fig.add_subplot(133)

min = SMEB[:, 0]
med = SMEB[:, 1]
max = SMEB[:, 2]

ax3.plot(yLT, min, c='0.8')
ax3.plot(yLT, med, c='k')
ax3.plot(yLT, max, c='0.8')

ax3.fill_between(yLT, min, med, color='0.8')
ax3.fill_between(yLT, med, max, color='0.8')

ax3.set_xlabel('Scaling Factor')
ax3.set_ylabel('Systems Match Score')
ax3.set_xlim(0, 1)
ax3.set_title('C', x=0)

plt.show()
