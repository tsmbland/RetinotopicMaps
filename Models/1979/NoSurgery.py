import time
import numpy as np
import sys
from joblib import Parallel, delayed
import Parameters as p

#################### OPTIONS ####################

nJobs = 1
Cores = 1


##################### ALGORITHM #################

def job(JobID):
    start = time.time()
    import Functions as f

    # Set Job Parameter(s)

    # Import Data
    Wpt = np.load('../../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
    f.Wpt[0, :, :, :, :] = Wpt[-1, :, :, :, :]
    f.Cpm = np.load('../../../RetinotopicMapsData/%s/RetinalConcentrations.npy' % ('{0:04}'.format(JobID)))
    Ctm = np.load('../../../RetinotopicMapsData/%s/TectalConcentrations.npy' % ('{0:04}'.format(JobID)))
    f.Ctm[:, 0, :, :] = Ctm[:, -1, :, :]

    # Model Type
    f.typestandard()

    f.setWtot()
    f.updateNc()
    f.normaliseCpm()
    f.normaliseCtm()

    # Iterations
    for iteration in range(1, f.Iterations + 1):
        f.Currentiteration += 1
        f.updateWeight()
        f.removesynapses()
        f.addsynapses()

        f.updateQtm()
        f.updatetectalconcs()
        f.normaliseCtm()

        f.updatexFieldcentres()

        sys.stdout.write('\r%i percent' % (iteration * 100 / f.Iterations))
        sys.stdout.flush()

    # Export Data
    f.savedata(JobID)

    # End
    sys.stdout.write('\rJob %s: Complete!' % ('{0:04}'.format(JobID)))
    sys.stdout.flush()
    end = time.time()
    elapsed = end - start
    print('\nTime elapsed: ', elapsed, 'seconds')


Parallel(n_jobs=Cores)(delayed(job)(JobID) for JobID in range(p.JobID, p.JobID + nJobs))


