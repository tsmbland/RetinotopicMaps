import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

################# OPTIONS ##################
sizechangethresh = 0.005
smthresh = 0.01

##################### IMPORT DATA #####################

JobID = int(input('JobID: '))
EB = input('Exclude Boundaries? (y/n) ')

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


################ CALCULATE STABILITY TIME #############

fieldsizechange = 1
systemsmatchchange = 1
iteration = 0
while (fieldsizechange > sizechangethresh or fieldsizechange < -sizechangethresh or
               systemsmatchchange > smthresh or systemsmatchchange < -smthresh) and iteration < len(
    Fieldsizechange):
    fieldsizechange = Fieldsizechange[iteration]
    systemsmatchchange = Systemsmatchchange[iteration]
    iteration += 1

iteration *= TRin

######################## PLOTS #######################
plt.subplot(2, 2, 1)
plt.title('Mean Receptive Field Separation')
plt.plot(range(TRin, len(Fieldseparation) * TRin, TRin), Fieldseparation[1:])
plt.ylabel('Mean Receptive Field Separation')
plt.xlabel('Time')
plt.xlim(0, len(Fieldseparation) * TRin)
plt.axvline(iteration, color='k', ls='--', lw='0.5')

plt.subplot(2, 2, 2)
plt.title('Receptive Field Separation Standard Deviation')
plt.plot(range(TRin, len(FieldseparationStdev) * TRin, TRin), FieldseparationStdev[1:])
plt.ylabel('Receptive Field Separation Standard Deviation')
plt.xlabel('Time')
plt.xlim(0, len(FieldseparationStdev) * TRin)
plt.axvline(iteration, color='k', ls='--', lw='0.5')

plt.subplot(2, 2, 3)
plt.title('Mean Receptive Field Size')
plt.plot(range(TRin, len(Fieldsize) * TRin, TRin), Fieldsize[1:])
plt.ylabel('Mean Receptive Field Diameter')
plt.xlabel('Time')
plt.xlim(0, len(Fieldsize) * TRin)
plt.axvline(iteration, color='k', ls='--', lw='0.5')

plt.subplot(2, 2, 4)
plt.title('Systems Match')
plt.plot(range(TRin, len(Systemsmatch) * TRin, TRin), Systemsmatch[1:])
plt.ylabel('Mean Distance Between Field Centre and Expected Field Centre')
plt.xlabel('Time')
plt.xlim(0, len(Systemsmatch) * TRin)
plt.axvline(iteration, color='k', ls='--', lw='0.5')

plt.show()
