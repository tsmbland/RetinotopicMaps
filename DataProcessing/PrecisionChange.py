import numpy as np
from joblib import Parallel, delayed

minJobID = int(input('Minimum JobID: '))
maxJobID = int(input('Maximum JobID: '))
Timecompression = int(input('Time Compression (1 = No Compression): '))
Cores = int(input('Cores: '))


def job(JobID):
    Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSize.npy' % ('{0:04}'.format(JobID)))
    Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparation.npy' % ('{0:04}'.format(JobID)))
    Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatch.npy' % ('{0:04}'.format(JobID)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

    Fieldsizechange = np.zeros([len(Fieldsize) - 1])
    Fieldseparationchange = np.zeros([len(Fieldseparation) - 1])
    Systemsmatchchange = np.zeros([len(Systemsmatch) - 1])

    # Iterations
    for iteration in range(len(Fieldsize) - 1):
        Fieldsizechange[iteration] = 100 * (Fieldsize[iteration + 1] - Fieldsize[iteration]) / (
            Fieldsize[iteration] * TRin)
        Fieldseparationchange[iteration] = 100 * (Fieldseparation[iteration + 1] - Fieldseparation[iteration]) / (
            Fieldseparation[iteration] * TRin)
        Systemsmatchchange[iteration] = 100 * (Systemsmatch[iteration + 1] - Systemsmatch[iteration]) / (
            Systemsmatch[iteration] * TRin)

    # Export Data
    np.save('../../RetinotopicMapsData/%s/FieldSizeChange' % ('{0:04}'.format(JobID)), Fieldsizechange)
    np.save('../../RetinotopicMapsData/%s/FieldSeparationChange' % ('{0:04}'.format(JobID)), Fieldseparationchange)
    np.save('../../RetinotopicMapsData/%s/SystemsMatchChange' % ('{0:04}'.format(JobID)), Systemsmatchchange)


Parallel(n_jobs=Cores)(delayed(job)(JobID) for JobID in range(minJobID, maxJobID + 1))
