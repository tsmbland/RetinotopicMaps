import numpy as np

###################### OPTIONS #########################

JobID = int(input('JobID: '))
TRin = np.load(
    '../../RetinotopicMapsData/%s/PrimaryTR.npy' % ('{0:04}'.format(JobID)))  # temporal resolution of input file
TRout = TRin  # temporal resolution of output file

#################### VARIABLES ##################

Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
xFieldCentres = np.load('../../RetinotopicMapsData/%s/xFieldCentres.npy' % ('{0:04}'.format(JobID)))

FieldCentres = np.zeros(
    [2, len(Weightmatrix[:, 0, 0, 0, 0]), len(Weightmatrix[0, :, 0, 0, 0]), len(Weightmatrix[0, 0, :, 0, 0])])
FieldSeparation = np.zeros(len(Weightmatrix[:, 0, 0, 0, 0]))
FieldSize = np.zeros(len(Weightmatrix[:, 0, 0, 0, 0]))
SystemsMatch = np.zeros(len(Weightmatrix[:, 0, 0, 0, 0]))


####################### FUNCTIONS #####################

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

    FieldCentres[0, i, :, :] = totaldim1 / weightsumdim1
    FieldCentres[1, i, :, :] = totaldim2 / weightsumdim2
    FieldCentres[:, i, :, :] = np.nan_to_num(FieldCentres[:, i, :, :])


def field_separation(i):
    totaldistance = 0
    count = 0

    # Field distance with closest neighbours in dim2
    for tdim1 in range(len(Weightmatrix[0, :, 0, 0, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim2 in range(len(Weightmatrix[0, 0, :, 0, 0])):
            if FieldCentres[0, i, tdim1, tdim2] != 0 and FieldCentres[1, i, tdim1, tdim2] != 0:
                fieldlistdim1.append(FieldCentres[0, i, tdim1, tdim2])
                fieldlistdim2.append(FieldCentres[1, i, tdim1, tdim2])
        for fieldcell in range(len(fieldlistdim1) - 1):
            totaldistance += np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2)
            count += 1

    # Field distance with closest neighbours in dim1
    for tdim2 in range(len(Weightmatrix[0, 0, :, 0, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim1 in range(len(Weightmatrix[0, :, 0, 0, 0])):
            if FieldCentres[0, i, tdim1, tdim2] != 0 and FieldCentres[1, i, tdim1, tdim2] != 0:
                fieldlistdim1.append(FieldCentres[0, i, tdim1, tdim2])
                fieldlistdim2.append(FieldCentres[1, i, tdim1, tdim2])
        for fieldcell in range(len(fieldlistdim1) - 1):
            totaldistance += np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2)
            count += 1

    meanseparation = totaldistance / count
    FieldSeparation[i] = meanseparation


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
                    rdim1 = len(Weightmatrix[0, 0, 0, :, 0]) - 1
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
                    rdim2 = len(Weightmatrix[0, 0, 0, 0, :]) - 1
                    weight = 0
                    while weight == 0:
                        weight = Weightmatrix[i, tdim1, tdim2, rdim1, rdim2]
                        rdim2 -= 1
                    max = rdim2 + 1
                    width = max - min
                area += width

            diameter = 2 * np.sqrt(area / (2 * np.pi))

            # Field size estimation
            totaldiameter += diameter
            count += 1

    # Mean field size estimation
    meandiameter = totaldiameter / count
    FieldSize[i] = meandiameter


def systems_match(i):
    totaldistance = 0
    count = 0

    for tdim1 in range(len(Weightmatrix[0, :, 0, 0, 0])):
        for tdim2 in range(len(Weightmatrix[0, 0, :, 0, 0])):
            if FieldCentres[0, i, tdim1, tdim2] != 0 and FieldCentres[1, i, tdim1, tdim2] != 0:
                totaldistance += np.sqrt((FieldCentres[0, i, tdim1, tdim2] - xFieldCentres[0, i, tdim1, tdim2]) ** 2 + (
                    FieldCentres[1, i, tdim1, tdim2] - xFieldCentres[1, i, tdim1, tdim2]) ** 2)
                count += 1

    meandistance = totaldistance / count
    SystemsMatch[i] = meandistance
