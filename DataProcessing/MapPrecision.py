import numpy as np
import sys
import time
import Functions as f
start = time.time()

###################### IMPORT DATA #####################
f.FieldCentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(f.JobID)))

###################### PRECISION MEASURES #####################
for i in range(0, len(f.Weightmatrix[:, 0, 0, 0, 0]), f.TRout // f.TRin):
    f.field_separation(i)
    f.field_size(i)
    f.systems_match(i)
    sys.stdout.write('\r%i percent' % (i * 100 / len(f.Weightmatrix[:, 0, 0, 0, 0])))
    sys.stdout.flush()

##################### EXPORT DATA #####################

np.save('../../RetinotopicMapsData/%s/FieldSize' % ('{0:04}'.format(f.JobID)), f.FieldSize)
np.save('../../RetinotopicMapsData/%s/FieldSeparation' % ('{0:04}'.format(f.JobID)), f.FieldSeparation)
np.save('../../RetinotopicMapsData/%s/SystemsMatch' % ('{0:04}'.format(f.JobID)), f.SystemsMatch)
np.save('../../RetinotopicMapsData/%s/SecondaryTR' % ('{0:04}'.format(f.JobID)), f.TRout)

###################### END ########################
sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
