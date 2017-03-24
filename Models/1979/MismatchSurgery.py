import time
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
    f.Cpm = np.load('../../../RetinotopicMapsData/%s/RetinalConcentrations.npy' % ('{0:04}'.format(JobID)))
    Ctm = np.load('../../../RetinotopicMapsData/%s/TectalConcentrations.npy' % ('{0:04}'.format(JobID)))
    f.Ctm[:, 0, :, :] = Ctm[:, -1, :, :]

    # Model Type
    f.typemismatchsurgery()

    # Initial Connections
    f.setWtot()
    f.initialconnections()

    f.updateNc()
    f.normaliseCpm()
    f.normaliseCtm()

    # Iterations
    for iteration in range(p.Iterations):
        f.updatetimepoint()

        f.updateWeight()
        f.removesynapses()
        f.addsynapses()

        f.updateQtm()
        f.updatetectalconcs()
        f.normaliseCtm()

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
