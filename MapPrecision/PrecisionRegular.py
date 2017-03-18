import sys
import time
from joblib import Parallel, delayed

minJobID = int(input('Minimum JobID: '))
maxJobID = int(input('Maximum JobID: '))
Cores = int(input('Cores: '))

def job(JobID):
    start = time.time()
    import Functions as f
    f.importdata(JobID, 1)

    # Iterations
    for i in range(0, len(f.FieldCentres[0, :, 0, 0]) // 1):
        f.updatetimepoint(1)
        f.MeanFieldSeparation(0)
        f.MeanFieldSize(0)
        f.MeanSystemsMatch(0)
        sys.stdout.write(
            '\rJob %s: %i percent' % (
            '{0:04}'.format(JobID), ((f.Currentiteration + 1) * 100 / len(f.FieldCentres[0, :, 0, 0]) * 1)))
        sys.stdout.flush()

    # Export Data
    f.savedata(JobID, 1)

    # End
    sys.stdout.write('\rJob %s Complete!' % ('{0:04}'.format(JobID)))
    sys.stdout.flush()
    end = time.time()
    elapsed = end - start
    print('\nTime elapsed: ', elapsed, 'seconds')

Parallel(n_jobs=Cores)(delayed(job)(JobID) for JobID in range(minJobID, maxJobID + 1))