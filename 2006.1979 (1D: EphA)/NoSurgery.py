import numpy as np
import time
import sys
import Functions as f

start = time.time()

################## IMPORT DATA ###################
Wpt = np.load('../Temporary Data/Weightmatrix.npy')
f.Wpt[0, :, :, :, :] = Wpt[-1, :, :, :, :]
Cta = np.load('../Temporary Data/EphrinA.npy')
f.Cta[0, :, :] = Cta[-1, :, :]

######################## ALGORITM #######################

# Model Type
f.typestandard()

f.setRetinalGradients()
f.updateNct()
f.setWtot()

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
np.save('../Temporary Data/EphrinA2', f.Cta[0:f.Iterations + 2:f.TRout, :, :])
np.save('../Temporary Data/xFieldcentres2', f.xFieldcentres[:, 0:f.Iterations + 2:f.TRout, :, :])

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
