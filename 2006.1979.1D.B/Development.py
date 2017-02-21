import numpy as np
import time
import sys
import Functions as f

start = time.time()

######################## ALGORITM #######################

# Model Type
f.typedevelopment()

# Set Gradients
f.setRetinalGradients()
f.setTectalGradients()
f.updateNct()

# Initial Connections
f.setWtot()
f.initialconnections()

# Iterations
for iteration in range(f.Iterations):
    f.updatetimepoint()

    f.updateWtot()
    f.updateWpt()
    f.removesynapses()
    f.addsynapses()

    f.growtectum()
    f.growretina()
    f.updateNct()

    f.updateI()
    f.updateCtb()

    f.updatexFieldcentres()

    sys.stdout.write('\r%i percent' % (iteration * 100 / f.Iterations))
    sys.stdout.flush()

#################### EXPORT DATA #################

np.save('../TemporaryData/Weightmatrix', f.Wpt)
np.save('../TemporaryData/EphrinB', f.Ctb)
np.save('../TemporaryData/xFieldcentres', f.xFieldcentres)

###################### END ########################

sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
