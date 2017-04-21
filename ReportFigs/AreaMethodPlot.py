import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

start = time.time()

##################### IMPORT DATA ########################

JobID = 813
Tdim1 = 25
Tdim2 = 25
Iteration = 500
print('Loading Data...')
Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
Fieldcentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(JobID)))
Fieldsizes = np.load('../../RetinotopicMapsData/%s/FieldSizes.npy' % ('{0:04}'.format(JobID)))
TRin1 = np.load('../../RetinotopicMapsData/%s/PrimaryTR.npy' % ('{0:04}'.format(JobID)))
TRin2 = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

######################## TABLE #########################

# Tabulate Weight Matrix
Table = np.zeros([len(Weightmatrix[0, 0, 0, :, 0]) * len(Weightmatrix[0, 0, 0, 0, :]), 3])

row = 0
for rdim1 in range(1, len(Weightmatrix[0, 0, 0, :, 0]) - 1):
    for rdim2 in range(1, len(Weightmatrix[0, 0, 0, 0, :]) - 1):
        if Weightmatrix[Iteration // TRin1, Tdim1, Tdim2, rdim1, rdim2] != 0.:
            Table[row, 0] = rdim1
            Table[row, 1] = rdim2
            Table[row, 2] = Weightmatrix[Iteration // TRin1, Tdim1, Tdim2, rdim1, rdim2]
            row += 1


####################### PLOT 1 ##########################

# Total Field
plt.subplot(121)
plt.xlim(23, 32)
plt.ylim(23, 32)
plt.ylabel('Dorsal - Ventral (j)')
plt.xlabel('Nasal - Temporal (i)')
plt.scatter(Table[:, 0], Table[:, 1], marker='s', c='k')

# Max
table = np.zeros([len(Weightmatrix[0, 0, 0, :, 0]) * len(Weightmatrix[0, 0, 0, 0, :]), 3])
row = 0
synapses = np.array(np.nonzero(Weightmatrix[Iteration // TRin1, Tdim1, Tdim2, :, :]))
for rdim1 in np.unique(synapses[0, :]):
    table[row, 0] = rdim1
    table[row, 1] = max(synapses[1, synapses[0, :] == rdim1])
    table[row, 2] = Weightmatrix[Iteration // TRin1, Tdim1, Tdim2, rdim1, max(synapses[1, synapses[0, :] == rdim1])]
    row += 1
plt.scatter(table[:, 0], table[:, 1], marker='s', c='g')

# Min
table = np.zeros([len(Weightmatrix[0, 0, 0, :, 0]) * len(Weightmatrix[0, 0, 0, 0, :]), 3])
row = 0
synapses = np.array(np.nonzero(Weightmatrix[Iteration // TRin1, Tdim1, Tdim2, :, :]))
for rdim1 in np.unique(synapses[0, :]):
    table[row, 0] = rdim1
    table[row, 1] = min(synapses[1, synapses[0, :] == rdim1])
    table[row, 2] = Weightmatrix[Iteration // TRin1, Tdim1, Tdim2, rdim1, min(synapses[1, synapses[0, :] == rdim1])]
    row += 1
plt.scatter(table[:, 0], table[:, 1], marker='s', c='r')

####################### PLOT 2 ##########################

# Total Field
plt.subplot(122)
plt.xlim(23, 32)
plt.ylim(23, 32)
plt.ylabel('Dorsal - Ventral (j)')
plt.xlabel('Nasal - Temporal (i)')
plt.scatter(Table[:, 0], Table[:, 1], marker='s', c='k')

# Max
table = np.zeros([len(Weightmatrix[0, 0, 0, :, 0]) * len(Weightmatrix[0, 0, 0, 0, :]), 3])
row = 0
synapses = np.array(np.nonzero(Weightmatrix[Iteration // TRin1, Tdim1, Tdim2, :, :]))
for rdim2 in np.unique(synapses[0, :]):
    table[row, 0] = max(synapses[0, synapses[1, :] == rdim2])
    table[row, 1] = rdim2
    table[row, 2] = Weightmatrix[Iteration // TRin1, Tdim1, Tdim2, max(synapses[0, synapses[1, :] == rdim2]), rdim2]
    row += 1
plt.scatter(table[:, 0], table[:, 1], marker='s', c='g')

# Min
table = np.zeros([len(Weightmatrix[0, 0, 0, :, 0]) * len(Weightmatrix[0, 0, 0, 0, :]), 3])
row = 0
synapses = np.array(np.nonzero(Weightmatrix[Iteration // TRin1, Tdim1, Tdim2, :, :]))
for rdim2 in np.unique(synapses[0, :]):
    table[row, 0] = min(synapses[0, synapses[1, :] == rdim2])
    table[row, 1] = rdim2
    table[row, 2] = Weightmatrix[Iteration // TRin1, Tdim1, Tdim2, min(synapses[0, synapses[1, :] == rdim2]), rdim2]
    row += 1
plt.scatter(table[:, 0], table[:, 1], marker='s', c='r')

plt.show()
