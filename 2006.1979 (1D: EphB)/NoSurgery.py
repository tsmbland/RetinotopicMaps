import numpy as np
import time
import sys
import Functions as f

start = time.time()

################## IMPORT DATA ###################
Wpt = np.load('../Temporary Data/Weightmatrix.npy')
f.Wpt[0, :, :, :, :] = Wpt[-1, :, :, :, :]
Ctb = np.load('../Temporary Data/EphrinB.npy')
f.Ctb[0, :, :] = Ctb[-1, :, :]

######################## ALGORITM #######################

# Model Type
f.typestandard()

f.setRetinalGradients()
f.updateNct()
f.setWtot()

# Iterations
for iteration in range(f.Iterations):
    f.updatetimepoint()

    f.updateWpt()
    f.removesynapses()
    f.addsynapses()

    f.updateI()
    f.updateCtb()

    f.updatexFieldcentres()

    sys.stdout.write('\r%i percent' % (iteration * 100 / f.Iterations))
    sys.stdout.flush()

#################### EXPORT DATA #################

np.save('../Temporary Data/Weightmatrix2', f.Wpt)
np.save('../Temporary Data/EphrinB2', f.Ctb)
np.save('../Temporary Data/xFieldcentres2', f.xFieldcentres)

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
