import numpy as np
import sys
import time
import Functions as f
start = time.time()

###################### FIELD CENTRES ######################
for i in range(0, len(f.Weightmatrix[:, 0, 0, 0, 0]), f.TRout // f.TRin):
    f.field_centre(i)
    sys.stdout.write('\r%i percent' % (i * 100 / len(f.Weightmatrix[:, 0, 0, 0, 0])))
    sys.stdout.flush()

##################### EXPORT DATA ###################

np.save('../../RetinotopicMapsData/%s/FieldCentres' % ('{0:04}'.format(f.JobID)), f.FieldCentres)
np.save('../../RetinotopicMapsData/%s/SecondaryTR' % ('{0:04}'.format(f.JobID)), f.TRout)

###################### END ########################
sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
