import matplotlib.pyplot as plt
import numpy as np

##################### IMPORT DATA #####################

JobID = int(input('JobID: '))

Fieldsizechange = np.load('../../RetinotopicMapsData/%s/FieldSizeChange.npy' % ('{0:04}'.format(JobID)))
Fieldseparationchange = np.load('../../RetinotopicMapsData/%s/FieldSeparationChange.npy' % ('{0:04}'.format(JobID)))
Systemsmatchchange = np.load('../../RetinotopicMapsData/%s/SystemsMatchChange.npy' % ('{0:04}'.format(JobID)))

###################### OPTIONS #######################

TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

######################## PLOTS #######################
plt.subplot(1, 3, 1)
plt.title('Receptive Field Separation')
plt.plot(range(TRin, len(Fieldseparationchange) * TRin, TRin), Fieldseparationchange[1:])
plt.ylabel('Mean Receptive Field Separation %Change')
plt.xlabel('Time')
plt.xlim(0, len(Fieldseparationchange) * TRin)

plt.subplot(1, 3, 2)
plt.title('Receptive Field Size')
plt.plot(range(TRin, len(Fieldsizechange) * TRin, TRin), Fieldsizechange[1:])
plt.ylabel('Mean Receptive Field Diameter %Change')
plt.xlabel('Time')
plt.xlim(0, len(Fieldsizechange) * TRin)

plt.subplot(1, 3, 3)
plt.title('Systems Match')
plt.plot(range(TRin, len(Systemsmatchchange) * TRin, TRin), Systemsmatchchange[1:])
plt.ylabel('Mean Distance Between Field Centre and Expected Field Centre %Change')
plt.xlabel('Time')
plt.xlim(0, len(Systemsmatchchange) * TRin)

plt.show()
