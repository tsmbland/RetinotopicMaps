import numpy as np
import sys
import time
start = time.time()

###################### IMPORT DATA #####################

Weightmatrix = np.load('../Temporary Data/Weightmatrix.npy')
Fieldcentres = np.load('../Temporary Data/Fieldcentres.npy')


###################### OPTIONS #########################

TRin = 5  # temporal resolution of input file
TRout = TRin  # temporal resolution of output files


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
    totaldiameter = 0
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

            diameter = 2*np.sqrt(area/(2*np.pi))

            # Field size estimation
            totaldiameter += diameter
            count += 1

    # Mean field size estimation
    meandiameter = totaldiameter / count
    Fieldsize[i] = meandiameter


for i in range(0, len(Weightmatrix[:, 0, 0, 0, 0]), TRout//TRin):
    field_separation(i)
    field_size(i)
    sys.stdout.write('\r%i percent' % (i * 100 / len(Weightmatrix[:, 0, 0, 0, 0])))
    sys.stdout.flush()


##################### EXPORT DATA #####################

np.save('../Temporary Data/Fieldsize', Fieldsize)
np.save('../Temporary Data/Fieldseparation', Fieldseparation)

###################### END ########################
sys.stdout.write('\rComplete!')
sys.stdout.flush()
end = time.time()
elapsed = end - start
print('\nTime elapsed: ', elapsed, 'seconds')