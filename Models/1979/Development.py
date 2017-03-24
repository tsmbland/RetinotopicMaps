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

    # Model Type
    f.typedevelopment()

    # Marker Locations
    f.setmarkerlocations()

    # Presynaptic Concentrations
    f.updateNc()
    f.setretinalconcs()
    f.normaliseCpm()

    # Initial Connections
    f.setWtot()
    f.initialconnections()

    # Initial Concentrations
    f.updateQtm()
    f.updatetectalconcs()
    f.normaliseCtm()

    # Iterations
    for iteration in range(1, f.Iterations + 1):
        f.Currentiteration += 1

        f.updateWtot()
        f.updateWeight()
        f.removesynapses()
        f.addsynapses()

        f.growtectum()
        f.growretina()
        f.updateNc()

        f.updateretinalconcs()
        f.normaliseCpm()

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
