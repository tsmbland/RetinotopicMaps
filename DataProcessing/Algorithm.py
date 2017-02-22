import numpy as np
import sys
import time
import Functions as f
start = time.time()


##################### ALGORITHM #######################

for i in range(0, len(f.Wpt[:, 0, 0, 0, 0]) * f.TRin // f.TRout):
    f.updatetimepoint()
    f.field_centre()
    f.field_separation()
    f.field_size()
    f.systems_match()
    sys.stdout.write('\r%i percent' % ((f.Currentiteration + 1) * 100 / len(f.Wpt[:, 0, 0, 0, 0]) * f.TRout // f.TRin))
    sys.stdout.flush()

##################### EXPORT DATA #####################

np.save('../../RetinotopicMapsData/%s/FieldCentres' % ('{0:04}'.format(f.JobID)), f.FieldCentres)
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