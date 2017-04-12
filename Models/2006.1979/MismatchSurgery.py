import numpy as np
import time
import sys
from joblib import Parallel, delayed
import Parameters as p

start = time.time()

#################### OPTIONS ####################

nJobs = 1
Cores = 1
inputJobID = int(input('Input JobID: '))

##################### ALGORITHM #################

def job(JobID):
    start = time.time()
    import Functions as f

    # Set Job Parameter(s)

    # Import Data
    Cta = np.load('../../../RetinotopicMapsData/%s/EphrinA.npy' % ('{0:04}'.format(inputJobID)))
    f.Cta[0, :, :] = Cta[-1, :, :]
    Ctb = np.load('../../../RetinotopicMapsData/%s/EphrinB.npy' % ('{0:04}'.format(inputJobID)))
    f.Ctb[0, :, :] = Ctb[-1, :, :]
    f.Cra = np.load('../../../RetinotopicMapsData/%s/EphA.npy' % ('{0:04}'.format(inputJobID)))
    f.Crb = np.load('../../../RetinotopicMapsData/%s/EphB.npy' % ('{0:04}'.format(inputJobID)))

    # Model Type
    f.typemismatchsurgery()
    f.updatexFieldcentres()

    f.updateNct()
    f.setWtot()
    f.initialconnections()

    # Iterations
    for iteration in range(p.Iterations):
        f.updatetimepoint()

        f.updateWpt()
        f.removesynapses()
        f.addsynapses()

        f.updateI()
        f.updateCta()
        f.updateCtb()

        f.updatexFieldcentres()

        sys.stdout.write('\rJob %s: %i percent' % ('{0:04}'.format(JobID), iteration * 100 / p.Iterations))
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