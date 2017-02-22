import numpy as np
import sys
import time

start = time.time()

###################### OPTIONS #########################

JobID = int(input('JobID: '))
TRin = np.load('../../RetinotopicMapsData/%s/PrimaryTR.npy' % ('{0:04}'.format(JobID)))  # temporal resolution of input file
TRout = TRin  # temporal resolution of output file

##################### IMPORT DATA ########################

Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))

###################### FIELD CENTRES ######################
Fieldcentres = np.zeros(
    [2, len(Weightmatrix[:, 0, 0, 0, 0]), len(Weightmatrix[0, :, 0, 0, 0]), len(Weightmatrix[0, 0, :, 0, 0])])


def field_centre(i):
    synapses = np.array(np.nonzero(Weightmatrix[i, :, :, :, :]))

    totaldim1 = np.zeros([len(Weightmatrix[0, :, 0, 0, 0]), len(Weightmatrix[0, 0, :, 0, 0])])
    totaldim2 = np.zeros([len(Weightmatrix[0, :, 0, 0, 0]), len(Weightmatrix[0, 0, :, 0, 0])])
    weightsumdim1 = np.zeros([len(Weightmatrix[0, :, 0, 0, 0]), len(Weightmatrix[0, 0, :, 0, 0])])
    weightsumdim2 = np.zeros([len(Weightmatrix[0, :, 0, 0, 0]), len(Weightmatrix[0, 0, :, 0, 0])])

    for synapse in range(int(len(synapses[0, :]))):
        tdim1 = synapses[0, synapse]
        tdim2 = synapses[1, synapse]
        rdim1 = synapses[2, synapse]
        rdim2 = synapses[3, synapse]
        weightsumdim1[tdim1, tdim2] += Weightmatrix[i, tdim1, tdim2, rdim1, rdim2]
        weightsumdim2[tdim1, tdim2] += Weightmatrix[i, tdim1, tdim2, rdim1, rdim2]
        totaldim1[tdim1, tdim2] += rdim1 * Weightmatrix[i, tdim1, tdim2, rdim1, rdim2]
        totaldim2[tdim1, tdim2] += rdim2 * Weightmatrix[i, tdim1, tdim2, rdim1, rdim2]

    Fieldcentres[0, i, :, :] = totaldim1 / weightsumdim1
    Fieldcentres[1, i, :, :] = totaldim2 / weightsumdim2
    Fieldcentres[:, i, :, :] = np.nan_to_num(Fieldcentres[:, i, :, :])


for i in range(0, len(Weightmatrix[:, 0, 0, 0, 0]), TRout // TRin):
    field_centre(i)
    sys.stdout.write('\r%i percent' % (i * 100 / len(Weightmatrix[:, 0, 0, 0, 0])))
    sys.stdout.flush()

##################### EXPORT DATA ###################

np.save('../../RetinotopicMapsData/%s/FieldCentres' % ('{0:04}'.format(JobID)), Fieldcentres)
np.save('../../RetinotopicMapsData/%s/SecondaryTR' % ('{0:04}'.format(JobID)), TRout)

###################### END ########################
sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
