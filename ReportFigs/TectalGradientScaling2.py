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



##################### PLOT ###################
fig = plt.figure()

# Field Separation Plot
ax1 = fig.add_subplot(131)

# Field Size Plot
ax2 = fig.add_subplot(132)

# Systems Match Plot
ax3 = fig.add_subplot(133)

################### IMPORT DATA ################

for column in range(11):
    for row in range(3):
        ID = JobID[column, row]

        # Import Data
        Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(ID)))
        Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(ID)))
        Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(ID)))
        TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(ID)))

        ax1.plot(range(0, len(Fieldseparation) * TRin, TRin), Fieldseparation)
        ax2.plot(range(TRin, len(Fieldsize) * TRin, TRin), Fieldsize[1:])
        ax3.plot(range(0, len(Systemsmatch) * TRin, TRin), Systemsmatch)


plt.show()
