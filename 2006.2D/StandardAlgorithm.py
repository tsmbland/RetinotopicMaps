# Standard model

import numpy as np
import sys
import time
import Functions as f

start = time.time()

######################## ALGORITM #######################


# Set Gradients
f.setRetinalGradients()
f.setTectalGradients()
f.updateNct()

# Initial Connections
f.initialconnections()

# Iterations
for iteration in range(f.Iterations):
    f.Currentiteration += 1

    f.updateI()
    f.updateCta()
    f.updateCtb()

    f.updateDpt()
    f.updateSpt()
    f.updateWpt()

    f.updatexFieldcentres()

    sys.stdout.write('\r%i percent' % (iteration * 100 / f.Iterations))
    sys.stdout.flush()

#################### EXPORT DATA #################

np.save('../TemporaryData/Weightmatrix', f.Wpt[0:f.Iterations + 2:f.TRout, :, :, :, :])
np.save('../TemporaryData/EphrinA', f.Cta[0:f.Iterations + 2:f.TRout, :, :])
np.save('../TemporaryData/EphrinB', f.Ctb[0:f.Iterations + 2:f.TRout, :, :])
np.save('../TemporaryData/xFieldCentres', f.xFieldcentres[:, 0:f.Iterations + 2:f.TRout, :, :])

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')