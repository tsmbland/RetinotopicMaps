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

        SepEB[column, row] = FieldseparationEB[0]
        SizeEB[column, row] = FieldsizeEB[1]
        SMEB[column, row] = SystemsmatchEB[0]

print('Separation: ', np.mean(np.mean(SepEB)))
print('Size: ', np.mean(np.mean(SizeEB)))
print('Systems Match: ', np.mean(np.mean(SMEB)))

