import numpy as np
import sys
import time

minJobID = int(input('Minimum JobID: '))
maxJobID = int(input('Maximum JobID: '))
Timecompression = int(input('Time Compression (1 = No Compression): '))

for JobID in range(minJobID, maxJobID + 1):
    start = time.time()
    import Functions as f

    f.importdata(JobID, Timecompression)
    TRout = f.TRin * Timecompression

    ##################### ALGORITHM #######################

    for i in range(0, len(f.Wpt[:, 0, 0, 0, 0]) // Timecompression):
        f.updatetimepoint(Timecompression)
        f.field_centre()
        f.field_separation()
        f.field_size()
        f.systems_match()
        sys.stdout.write(
            '\rJob %s: %i percent' % (
            '{0:04}'.format(JobID), ((f.Currentiteration + 1) * 100 / len(f.Wpt[:, 0, 0, 0, 0]) * Timecompression)))
        sys.stdout.flush()

    ##################### EXPORT DATA #####################

    np.save('../../RetinotopicMapsData/%s/FieldCentres' % ('{0:04}'.format(JobID)), f.FieldCentres)
    np.save('../../RetinotopicMapsData/%s/FieldSize' % ('{0:04}'.format(JobID)), f.FieldSize)
    np.save('../../RetinotopicMapsData/%s/FieldSeparation' % ('{0:04}'.format(JobID)), f.FieldSeparation)
    np.save('../../RetinotopicMapsData/%s/SystemsMatch' % ('{0:04}'.format(JobID)), f.SystemsMatch)
    np.save('../../RetinotopicMapsData/%s/SecondaryTR' % ('{0:04}'.format(JobID)), f.Currentiteration)

    ###################### END ########################
    sys.stdout.write('\rJob %s Complete!' % ('{0:04}'.format(JobID)))
    sys.stdout.flush()
    end = time.time()
    elapsed = end - start
    print('\nTime elapsed: ', elapsed, 'seconds')
