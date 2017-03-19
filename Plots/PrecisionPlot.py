import matplotlib.pyplot as plt
import numpy as np

##################### IMPORT DATA #####################

JobID = int(input('JobID: '))
EB = input('Exclude Boundaries? (y/n) ')

if EB == 'n':
    Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSize.npy' % ('{0:04}'.format(JobID)))
    Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparation.npy' % ('{0:04}'.format(JobID)))
    Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatch.npy' % ('{0:04}'.format(JobID)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

elif EB == 'y':
    Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(JobID)))
    Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(JobID)))
    Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(JobID)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

######################## PLOTS #######################
plt.subplot(1, 3, 1)
plt.title('Receptive Field Separation')
plt.plot(range(TRin, len(Fieldseparation) * TRin, TRin), Fieldseparation[1:])
plt.ylabel('Mean Receptive Field Separation')
plt.xlabel('Time')
plt.xlim(0, len(Fieldseparation) * TRin)

plt.subplot(1, 3, 2)
plt.title('Receptive Field Size')
plt.plot(range(TRin, len(Fieldsize) * TRin, TRin), Fieldsize[1:])
plt.ylabel('Mean Receptive Field Diameter')
plt.xlabel('Time')
plt.xlim(0, len(Fieldsize) * TRin)

plt.subplot(1, 3, 3)
plt.title('Systems Match')
plt.plot(range(TRin, len(Systemsmatch) * TRin, TRin), Systemsmatch[1:])
plt.ylabel('Mean Distance Between Field Centre and Expected Field Centre')
plt.xlabel('Time')
plt.xlim(0, len(Systemsmatch) * TRin)

plt.show()
