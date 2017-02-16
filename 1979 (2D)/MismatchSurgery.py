import numpy as np
import time
import sys
import Functions as f

start = time.time()

################## IMPORT DATA ###################
f.Cpm = np.load('../Temporary Data/Retinal Concentrations.npy')
Ctm = np.load('../Temporary Data/Tectal Concentrations.npy')
f.Ctm[:, 0, :, :] = Ctm[:, -1, :, :]

################## ALGORITHM ######################

# MISMATCH SURGERY
f.typemismatchsurgery()

# INITIAL CONNECTIONS
f.setWtot()
f.initialconnections()

f.updateNc()
f.normaliseCpm()
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

np.save('../Temporary Data/Weightmatrix2', f.Wpt[0:f.Iterations + 2:f.TRout, :, :, :, :])
np.save('../Temporary Data/Retinal Concentrations2', f.Cpm)
np.save('../Temporary Data/Tectal Concentrations2', f.Ctm[:, 0:f.Iterations + 2:f.TRout, :, :])
np.save('../Temporary Data/xFieldcentres2', f.xFieldcentres[:, 0:f.Iterations + 2:f.TRout, :, :])

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
