import numpy as np
import sys
import matplotlib.pyplot as plt
import time
start = time.time()

###################### IMPORT DATA #####################

Weightmatrix = np.load('../Temporary Data/Weightmatrix.npy')
Fieldcentres = np.load('../Temporary Data/Fieldcentres.npy')



###################### PRECISION MEASURES #####################

Fieldseparation = np.zeros(len(Weightmatrix[:, 0, 0, 0, 0]))
Fieldsize = np.zeros(len(Weightmatrix[:, 0, 0, 0, 0]))


def field_separation(i):
    totaldistance = 0
    count = 0

    # Field distance with closest neighbours in dim2
    for tdim1 in range(len(Weightmatrix[0, :, 0, 0, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim2 in range(len(Weightmatrix[0, 0, :, 0, 0])):
            if Fieldcentres[0, i, tdim1, tdim2] != 0 and Fieldcentres[1, i, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, i, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, i, tdim1, tdim2])
        for fieldcell in range(len(fieldlistdim1) - 1):
            totaldistance += np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2)
            count += 1

    # Field distance with closest neighbours in dim1
    for tdim2 in range(len(Weightmatrix[0, 0, :, 0, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim1 in range(len(Weightmatrix[0, :, 0, 0, 0])):
            if Fieldcentres[0, i, tdim1, tdim2] != 0 and Fieldcentres[1, i, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, i, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, i, tdim1, tdim2])
        for fieldcell in range(len(fieldlistdim1) - 1):
            totaldistance += np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2)
            count += 1

    meanseparation = totaldistance / count
    Fieldseparation[i] = meanseparation


def field_size(i):
    totalarea = 0
    count = 0
    for tdim1 in range(len(Weightmatrix[0, :, 0, 0, 0])):
        for tdim2 in range(len(Weightmatrix[0, 0, :, 0, 0])):

            area = 0
            # Scanning in dim1
            for rdim2 in range(len(Weightmatrix[0, 0, 0, 0, :])):
                width = 0
                rdim1 = 0
                weight = 0
                while weight == 0 and rdim1 < len(Weightmatrix[0, 0, 0, :, 0]):
                    weight = Weightmatrix[i, tdim1, tdim2, rdim1, rdim2]
                    rdim1 += 1
                min = rdim1 - 1
                if weight != 0:
                    rdim1 = len(Weightmatrix[0, 0, 0, :, 0])-1
                    weight = 0
                    while weight == 0:
                        weight = Weightmatrix[i, tdim1, tdim2, rdim1, rdim2]
                        rdim1 -= 1
                    max = rdim1 + 1
                    width = max - min
                area += width

            # Scanning in dim2
            for rdim1 in range(len(Weightmatrix[0, 0, 0, :, 0])):
                width = 0
                rdim2 = 0
                weight = 0
                while weight == 0 and rdim2 < len(Weightmatrix[0, 0, 0, 0, :]):
                    weight = Weightmatrix[i, tdim1, tdim2, rdim1, rdim2]
                    rdim2 += 1
                min = rdim2 - 1
                if weight != 0:
                    rdim2 = len(Weightmatrix[0, 0, 0, 0, :])-1
                    weight = 0
                    while weight == 0:
                        weight = Weightmatrix[i, tdim1, tdim2, rdim1, rdim2]
                        rdim2 -= 1
                    max = rdim2 + 1
                    width = max - min
                area += width

            # Field size estimation
            totalarea += area / 2
            count += 1

    # Mean field size estimation
    meanarea = totalarea / count
    Fieldsize[i] = meanarea


for i in range(len(Weightmatrix[:, 0, 0, 0, 0])):
    field_separation(i)
    field_size(i)
    sys.stdout.write('\rProcessing data... %i percent' % (i * 100 / len(Weightmatrix[:, 0, 0, 0, 0])))
    sys.stdout.flush()

######################## PLOTS #######################
plt.subplot(1, 2, 1)
plt.title('Receptive Field Separation')
plt.plot(Fieldseparation)
plt.ylabel('Mean Receptive Field Separation')
plt.xlabel('Time')

plt.subplot(1, 2, 2)
plt.title('Receptive Field Size')
plt.plot(Fieldsize)
plt.ylabel('Mean Receptive Field Area')
plt.xlabel('Time')

###################### END ########################
sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')
params = {'font.size': '10'}
plt.rcParams.update(params)
plt.show()
