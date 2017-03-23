import numpy as np
import time
import sys
from joblib import Parallel, delayed
import Parameters as p

start = time.time()

#################### OPTIONS ####################

nJobs = 1
Cores = 1


##################### ALGORITHM #################

def job(JobID):
    start = time.time()
    import Functions as f

    # Set Job Parameter(s)

    # Import Data
    Cta = np.load('../../../RetinotopicMapsData/%s/EphrinA.npy' % ('{0:04}'.format(JobID)))
    f.Cta[0, :, :] = Cta[-1, :, :]
    Ctb = np.load('../../../RetinotopicMapsData/%s/EphrinB.npy' % ('{0:04}'.format(JobID)))
    f.Ctb[0, :, :] = Ctb[-1, :, :]

    # Model Type
    f.typemismatchsurgery()

    f.setRetinalGradients()
    f.updateNct()
    f.setWtot()
    f.initialconnections()

    # Iterations
    for iteration in range(f.Iterations):
        f.updatetimepoint()

        f.updateWpt()
        f.removesynapses()
        f.addsynapses()

        f.updateI()
        f.updateCta()
        f.updateCtb()

        f.updatexFieldcentres()

        sys.stdout.write('\r%i percent' % (iteration * 100 / f.Iterations))
        sys.stdout.flush()

    # Export Data
    f.savedata()

    # End
    sys.stdout.write('\rComplete!')
    sys.stdout.flush()
    end = time.time()
    elapsed = end - start
    print('\nTime elapsed: ', elapsed, 'seconds')


Parallel(n_jobs=Cores)(delayed(job)(JobID) for JobID in range(p.JobID, p.JobID + nJobs))