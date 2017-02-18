import numpy as np
import time
import sys
import Functions as f

start = time.time()


################## IMPORT DATA ###################
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
    f.Currentiteration += 1

    f.updateI()
    f.updateCta()

    f.updateWpt()
    f.removesynapses()
    f.addsynapses()

    f.updatexFieldcentres()

    sys.stdout.write('\r%i percent' % (iteration * 100 / f.Iterations))
    sys.stdout.flush()

#################### EXPORT DATA #################

np.save('../Temporary Data/Weightmatrix2', f.Wpt[0:f.Iterations + 2:f.TRout, :, :, :, :])
np.save('../Temporary Data/EphrinB2', f.Ctb[0:f.Iterations + 2:f.TRout, :, :])
np.save('../Temporary Data/xFieldcentres2', f.xFieldcentres[:, 0:f.Iterations + 2:f.TRout, :, :])

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
