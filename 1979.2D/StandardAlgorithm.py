import numpy as np
import time
import sys
import Functions as f

start = time.time()

######################## ALGORITHM ##########################

# MODEL TYPE

f.typestandard()

# MARKER LOCATIONS

f.setmarkerlocations()

# PRESYNAPTIC CONCENTRATIONS

f.updateNc()
f.setretinalconcs()
f.normaliseCpm()

# INITIAL CONNECTIONS

f.setWtot()
f.initialconnections()

# INITIAL CONCENTRATIONS

f.updateQtm()
f.updatetectalconcs()
f.normaliseCtm()

# ITERATIONS

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

#################### EXPORT DATA #################

np.save('../TemporaryData/Weightmatrix', f.Wpt[0:f.Iterations + 2:f.TRout, :, :, :, :])
np.save('../TemporaryData/RetinalConcentrations', f.Cpm)
np.save('../TemporaryData/TectalConcentrations', f.Ctm[:, 0:f.Iterations + 2:f.TRout, :, :])
np.save('../TemporaryData/xFieldCentres', f.xFieldcentres[:, 0:f.Iterations + 2:f.TRout, :, :])

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
