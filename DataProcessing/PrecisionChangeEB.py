import numpy as np
from joblib import Parallel, delayed

minJobID = int(input('Minimum JobID: '))
maxJobID = int(input('Maximum JobID: '))
Timecompression = int(input('Time Compression (1 = No Compression): '))
Cores = int(input('Cores: '))


def job(JobID):
    Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(JobID)))
    Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(JobID)))
    Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(JobID)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTREB.npy' % ('{0:04}'.format(JobID)))

    Fieldsizechange = np.zeros([len(Fieldsize) - 1])
    Fieldseparationchange = np.zeros([len(Fieldseparation) - 1])
    Systemsmatchchange = np.zeros([len(Systemsmatch) - 1])

    # Iterations
    for iteration in range(len(Fieldsize) - 1):
        Fieldsizechange[iteration] = (Fieldsize[iteration + 1] - Fieldsize[iteration]) / TRin
        Fieldseparationchange[iteration] = (Fieldseparation[iteration + 1] - Fieldseparation[iteration]) / TRin
        Systemsmatchchange[iteration] = (Systemsmatch[iteration + 1] - Systemsmatch[iteration]) / TRin

    # Export Data
    np.save('../../RetinotopicMapsData/%s/FieldSizeChangeEB' % ('{0:04}'.format(JobID)), Fieldsizechange)
    np.save('../../RetinotopicMapsData/%s/FieldSeparationChangeEB' % ('{0:04}'.format(JobID)), Fieldseparationchange)
    np.save('../../RetinotopicMapsData/%s/SystemsMatchChangeEB' % ('{0:04}'.format(JobID)), Systemsmatchchange)


Parallel(n_jobs=Cores)(delayed(job)(JobID) for JobID in range(minJobID, maxJobID + 1))
