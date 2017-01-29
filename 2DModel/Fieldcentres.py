import numpy as np
import sys
import time

start = time.time()

##################### IMPORT DATA ########################

Weightmatrix = np.load('Weightmatrix.npy')

###################### FIELD CENTRES ######################
Fieldcentres = np.zeros(
    [2, len(Weightmatrix[:, 0, 0, 0, 0]), len(Weightmatrix[0, :, 0, 0, 0]), len(Weightmatrix[0, 0, :, 0, 0])])


def field_centre(i):
    totaldim1 = np.zeros([len(Weightmatrix[0, :, 0, 0, 0]), len(Weightmatrix[0, 0, :, 0, 0])])
    totaldim2 = np.zeros([len(Weightmatrix[0, :, 0, 0, 0]), len(Weightmatrix[0, 0, :, 0, 0])])
    weightsumdim1 = np.zeros([len(Weightmatrix[0, :, 0, 0, 0]), len(Weightmatrix[0, 0, :, 0, 0])])
    weightsumdim2 = np.zeros([len(Weightmatrix[0, :, 0, 0, 0]), len(Weightmatrix[0, 0, :, 0, 0])])

    for tdim1 in range(len(Weightmatrix[0, :, 0, 0, 0])):
        for tdim2 in range(len(Weightmatrix[0, 0, :, 0, 0])):
            weightsumdim1[tdim1, tdim2] = sum(sum(Weightmatrix[i, tdim1, tdim2, :, :]))
            weightsumdim2[tdim1, tdim2] = sum(sum(Weightmatrix[i, tdim1, tdim2, :, :]))
            for rdim1 in range(len(Weightmatrix[0, 0, 0, :, 0])):
                for rdim2 in range(len(Weightmatrix[0, 0, 0, 0, :])):
                    totaldim1[tdim1, tdim2] += rdim1 * Weightmatrix[i, tdim1, tdim2, rdim1, rdim2]
                    totaldim2[tdim1, tdim2] += rdim2 * Weightmatrix[i, tdim1, tdim2, rdim1, rdim2]

    Fieldcentres[0, i, :, :] = totaldim1 / weightsumdim1
    Fieldcentres[1, i, :, :] = totaldim2 / weightsumdim2


for i in range(len(Weightmatrix[:, 0, 0, 0, 0])):
    field_centre(i)
    sys.stdout.write('\r%i percent' % (i * 100 / len(Weightmatrix[:, 0, 0, 0, 0])))
    sys.stdout.flush()

##################### EXPORT DATA ###################

np.save('Fieldcentres', Fieldcentres)

###################### END ########################
sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
