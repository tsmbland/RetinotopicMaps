import numpy as np
import time
import sys
import Functions as f

start = time.time()

################## IMPORT DATA ###################
Cta = np.load('../Temporary Data/EphrinA.npy')
f.Cta[0, :, :] = Cta[-1, :, :]
Ctb = np.load('../Temporary Data/EphrinB.npy')
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

    f.updateWpt()
    f.removesynapses()
    f.addsynapses()

    f.updateI()
    f.updateCta()
    f.updateCtb()

    f.updatexFieldcentres()

    sys.stdout.write('\r%i percent' % (iteration * 100 / f.Iterations))
    sys.stdout.flush()

#################### EXPORT DATA #################

np.save('../Temporary Data/Weightmatrix2', f.Wpt)
np.save('../Temporary Data/EphrinA2', f.Cta)
np.save('../Temporary Data/EphrinB2', f.Ctb)
np.save('../Temporary Data/xFieldcentres2', f.xFieldcentres)

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
