import numpy as np
import time
import sys
import Functions as f

start = time.time()


################## IMPORT DATA ###################
Ctb = np.load('../TemporaryData/EphrinB.npy')
f.Ctb[0, :, :] = Ctb[-1, :, :]


######################## ALGORITM #######################

# Model Type
f.typemismatchsurgery()

# Set Gradients
f.setRetinalGradients()
f.updateNct()

# Initial Connections
f.setWtot()
f.initialconnections()

# Iterations
for iteration in range(f.Iterations):
    f.updatetimepoint()

    f.updateI()
    f.updateCtb()

    f.updateWpt()
    f.removesynapses()
    f.addsynapses()

    f.updatexFieldcentres()

    sys.stdout.write('\r%i percent' % (iteration * 100 / f.Iterations))
    sys.stdout.flush()

#################### EXPORT DATA #################

np.save('../TemporaryData/Weightmatrix2', f.Wpt)
np.save('../TemporaryData/EphrinB2', f.Ctb)
np.save('../TemporaryData/xFieldcentres2', f.xFieldcentres)

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
