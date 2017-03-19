import matplotlib.pyplot as plt
import numpy as np

##################### IMPORT DATA #####################

JobID = int(input('JobID: '))
EB = input('Exclude Boundaries? (y/n) ')

if EB == 'n':
    Fieldsizechange = np.load('../../RetinotopicMapsData/%s/FieldSizeChange.npy' % ('{0:04}'.format(JobID)))
    Fieldseparationchange = np.load('../../RetinotopicMapsData/%s/FieldSeparationChange.npy' % ('{0:04}'.format(JobID)))
    Systemsmatchchange = np.load('../../RetinotopicMapsData/%s/SystemsMatchChange.npy' % ('{0:04}'.format(JobID)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

elif EB == 'y':
    Fieldsizechange = np.load('../../RetinotopicMapsData/%s/FieldSizeChangeEB.npy' % ('{0:04}'.format(JobID)))
    Fieldseparationchange = np.load(
        '../../RetinotopicMapsData/%s/FieldSeparationChangeEB.npy' % ('{0:04}'.format(JobID)))
    Systemsmatchchange = np.load('../../RetinotopicMapsData/%s/SystemsMatchChangeEB.npy' % ('{0:04}'.format(JobID)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

Iteration = int(input('Iteration: '))

#################### PRINT DATA #######################

print('Field Separation Change: ', Fieldseparationchange[int(Iteration / TRin)])
print('Field Size Change: ', Fieldsizechange[int(Iteration / TRin)])
print('Systems Match Change: ', Systemsmatchchange[int(Iteration / TRin)])
