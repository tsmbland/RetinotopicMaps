import matplotlib.pyplot as plt
import numpy as np

##################### IMPORT DATA #####################

JobID = int(input('JobID: '))

Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSize.npy' % ('{0:04}'.format(JobID)))
Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparation.npy' % ('{0:04}'.format(JobID)))
Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatch.npy' % ('{0:04}'.format(JobID)))
TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

Iteration = int(input('Iteration: '))



#################### PRINT DATA #######################

print('Field Separation: ', Fieldseparation[int(Iteration/TRin)])
print('Field Size: ', Fieldsize[int(Iteration/TRin)])
print('Systems Match: ', Systemsmatch[int(Iteration/TRin)])