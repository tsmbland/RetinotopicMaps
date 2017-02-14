# Fixed tectal gradients

import numpy as np
import time
import sys
import Functions as f

start = time.time()

######################## ALGORITM #######################


# Set Gradients
f.setRetinalGradients()
f.setTectalGradients()

# Initial Connections
f.initialconnections1()

# Iterations
for iteration in range(f.Iterations):
    f.Currentiteration += 1

    f.updateDpt()
    f.updateSpt()
    f.updateWpt1()

    sys.stdout.write('\r%i percent' % (iteration * 100 / f.Iterations))
    sys.stdout.flush()

#################### EXPORT DATA #################

np.save('../Temporary Data/Weightmatrix', f.Wpt[0:f.Iterations + 2:f.TRout, :, :, :, :])
np.save('../Temporary Data/EphrinA', f.Cta[0:f.Iterations + 2:f.TRout, :, :])
np.save('../Temporary Data/EphrinB', f.Ctb[0:f.Iterations + 2:f.TRout, :, :])

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
