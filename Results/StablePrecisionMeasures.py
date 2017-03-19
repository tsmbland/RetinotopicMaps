import matplotlib.pyplot as plt
import numpy as np

##################### OPTIONS ########################
sizechangethresh = 0.005
smthresh = 0.01

##################### IMPORT DATA #####################

JobID = int(input('JobID: '))
EB = input('Exclude Boundaries? (y/n) ')

if EB == 'n':
    Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSize.npy' % ('{0:04}'.format(JobID)))
    Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparation.npy' % ('{0:04}'.format(JobID)))
    Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatch.npy' % ('{0:04}'.format(JobID)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

    Fieldsizechange = np.load('../../RetinotopicMapsData/%s/FieldSizeChange.npy' % ('{0:04}'.format(JobID)))
    Fieldseparationchange = np.load('../../RetinotopicMapsData/%s/FieldSeparationChange.npy' % ('{0:04}'.format(JobID)))
    Systemsmatchchange = np.load('../../RetinotopicMapsData/%s/SystemsMatchChange.npy' % ('{0:04}'.format(JobID)))

    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

if EB == 'y':
    Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(JobID)))
    Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(JobID)))
    Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(JobID)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTREB.npy' % ('{0:04}'.format(JobID)))

    Fieldsizechange = np.load('../../RetinotopicMapsData/%s/FieldSizeChangeEB.npy' % ('{0:04}'.format(JobID)))
    Fieldseparationchange = np.load(
        '../../RetinotopicMapsData/%s/FieldSeparationChangeEB.npy' % ('{0:04}'.format(JobID)))
    Systemsmatchchange = np.load('../../RetinotopicMapsData/%s/SystemsMatchChangeEB.npy' % ('{0:04}'.format(JobID)))

    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

#################### CALCULATE TIME #######################

fieldsizechange = 1
systemsmatchchange = 1
iteration = 0
while (fieldsizechange > sizechangethresh or fieldsizechange < -sizechangethresh or
               systemsmatchchange > smthresh or systemsmatchchange < -smthresh) and iteration < len(Fieldsizechange):
    fieldsizechange = Fieldsizechange[iteration]
    systemsmatchchange = Systemsmatchchange[iteration]
    iteration += 1

##################### PRINT DATA ##########################

print('Stability Time: ', iteration * TRin)
print('Field Separation: ', Fieldseparation[int(iteration)])
print('Field Size: ', Fieldsize[int(iteration)])
print('Systems Match: ', Systemsmatch[int(iteration)])
