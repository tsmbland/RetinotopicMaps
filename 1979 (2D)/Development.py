import numpy as np
import time
import sys
import Functions as f

start = time.time()

######################## ALGORITHM ##########################

# MODEL TYPE
f.typedevelopment()

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

#################### EXPORT DATA #################

np.save('../Temporary Data/Weightmatrix', f.Wpt[0:f.Iterations + 2:f.TRout, :, :, :, :])
np.save('../Temporary Data/Retinal Concentrations', f.Cpm)
np.save('../Temporary Data/Tectal Concentrations', f.Ctm[:, 0:f.Iterations + 2:f.TRout, :, :])
np.save('../Temporary Data/xFieldcentres', f.xFieldcentres[:, 0:f.Iterations + 2:f.TRout, :, :])

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
