import sys
import time
import numpy as np
from joblib import Parallel, delayed

minJobID = int(input('Minimum JobID: '))
maxJobID = int(input('Maximum JobID: '))
Cores = int(input('Cores: '))
border = 5

################ PRECISION ###################

def job(JobID):
    start = time.time()
    import Functions as f
    f.importdata(JobID, 1)

    # Iterations
    for i in range(0, len(f.FieldCentres[0, :, 0, 0]) // 1):
        f.updatetimepoint(1)
        f.MeanFieldSeparation()
        f.MeanFieldSize()
        f.MeanSystemsMatch()
        f.MeanFieldSeparationEB(border)
        f.MeanFieldSizeEB(border)
        f.MeanSystemsMatchEB(border)
        sys.stdout.write(
            '\rJob %s: %i percent' % (
            '{0:04}'.format(JobID), ((f.Currentiteration + 1) * 100 / len(f.FieldCentres[0, :, 0, 0]) * 1)))
        sys.stdout.flush()

    # Export Data
    f.savedata(JobID, 1)
    f.savedataEB(JobID, 1)

    # End
    sys.stdout.write('\rJob %s Complete!' % ('{0:04}'.format(JobID)))
    sys.stdout.flush()
    end = time.time()
    elapsed = end - start
    print('\nTime elapsed: ', elapsed, 'seconds')

Parallel(n_jobs=Cores)(delayed(job)(JobID) for JobID in range(minJobID, maxJobID + 1))


################ PRECISION CHANGE ###################

def PrecisionChange(JobID):
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

    fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSize.npy' % ('{0:04}'.format(JobID)))
    fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparation.npy' % ('{0:04}'.format(JobID)))
    systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatch.npy' % ('{0:04}'.format(JobID)))

    fieldsizeEB = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(JobID)))
    fieldseparationEB = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(JobID)))
    systemsmatchEB = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(JobID)))

    fieldsizechange = np.zeros([len(fieldsize) - 1])
    fieldseparationchange = np.zeros([len(fieldseparation) - 1])
    systemsmatchchange = np.zeros([len(systemsmatch) - 1])

    fieldsizechangeEB = np.zeros([len(fieldsize) - 1])
    fieldseparationchangeEB = np.zeros([len(fieldseparation) - 1])
    systemsmatchchangeEB = np.zeros([len(systemsmatch) - 1])

    # Iterations
    for iteration in range(len(fieldsize) - 1):
        fieldsizechange[iteration] = 100 * (fieldsize[iteration + 1] - fieldsize[iteration]) / (
            fieldsize[iteration] * TRin)
        fieldseparationchange[iteration] = 100 * (fieldseparation[iteration + 1] - fieldseparation[iteration]) / (
            fieldseparation[iteration] * TRin)
        systemsmatchchange[iteration] = 100 * (systemsmatch[iteration + 1] - systemsmatch[iteration]) / (
            systemsmatch[iteration] * TRin)

        fieldsizechangeEB[iteration] = 100 * (fieldsizeEB[iteration + 1] - fieldsizeEB[iteration]) / (
            fieldsizeEB[iteration] * TRin)
        fieldseparationchangeEB[iteration] = 100 * (fieldseparationEB[iteration + 1] - fieldseparationEB[iteration]) / (
            fieldseparationEB[iteration] * TRin)
        systemsmatchchangeEB[iteration] = 100 * (systemsmatchEB[iteration + 1] - systemsmatchEB[iteration]) / (
            systemsmatchEB[iteration] * TRin)

    # Export Data
    np.save('../../RetinotopicMapsData/%s/FieldSizeChange' % ('{0:04}'.format(JobID)), fieldsizechange)
    np.save('../../RetinotopicMapsData/%s/FieldSeparationChange' % ('{0:04}'.format(JobID)), fieldseparationchange)
    np.save('../../RetinotopicMapsData/%s/SystemsMatchChange' % ('{0:04}'.format(JobID)), systemsmatchchange)

    np.save('../../RetinotopicMapsData/%s/FieldSizeChangeEB' % ('{0:04}'.format(JobID)), fieldsizechangeEB)
    np.save('../../RetinotopicMapsData/%s/FieldSeparationChangeEB' % ('{0:04}'.format(JobID)), fieldseparationchangeEB)
    np.save('../../RetinotopicMapsData/%s/SystemsMatchChangeEB' % ('{0:04}'.format(JobID)), systemsmatchchangeEB)

Parallel(n_jobs=Cores)(delayed(PrecisionChange)(JobID) for JobID in range(minJobID, maxJobID + 1))