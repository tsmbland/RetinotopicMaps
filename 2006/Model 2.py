# 2006/1979 hybrid model

import numpy as np
import time
import sys
import Functions as f

start = time.time()

######################## ALGORITM #######################


# Set Gradients
f.setRetinalGradients()
f.setTectalGradients()
f.updateNct()

# Initial Connections
for rdim1 in range(f.Rmindim1, f.Rmaxdim1 + 1):
    for rdim2 in range(f.Rmindim2, f.Rmaxdim2 + 1):
        f.initialconections2(rdim1, rdim2)

# Iterations
for iteration in range(f.Iterations):
    f.Currentiteration += 1

    f.updateI()
    f.updateCta()
    f.updateCtb()

    f.updateWpt2()
    f.removesynapses()
    f.addsynapses()

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
