import numpy as np


####################### FUNCTIONS #####################

def importdata(jobid, timecompression):
    global Wpt, xFieldCentres, FieldCentres, FieldSeparation, FieldSize, SystemsMatch, MeanChange, TRin, Currentiteration, Weightiteration
    Wpt = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(jobid)))
    xFieldCentres = np.load('../../RetinotopicMapsData/%s/xFieldCentres.npy' % ('{0:04}'.format(jobid)))
    TRin = np.load('../../RetinotopicMapsData/%s/PrimaryTR.npy' % ('{0:04}'.format(jobid)))
    FieldCentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(jobid)))

    FieldSeparation = np.zeros(int(len(Wpt[:, 0, 0, 0, 0]) / timecompression))
    FieldSize = np.zeros(int(len(Wpt[:, 0, 0, 0, 0]) / timecompression))
    SystemsMatch = np.zeros(int(len(Wpt[:, 0, 0, 0, 0]) / timecompression))

    Currentiteration = -1
    Weightiteration = -1


def updatetimepoint(timecompression):
    global Currentiteration, Weightiteration
    Currentiteration += 1
    Weightiteration += timecompression


def field_separation(border):
    totaldistance = 0
    count = 0

    # Field distance with closest neighbours in dim2
    for tdim1 in range(border, len(Wpt[0, :, 0, 0, 0]) - border):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim2 in range(border, len(Wpt[0, 0, :, 0, 0]) - border):
            if FieldCentres[0, Currentiteration, tdim1, tdim2] != 0 and FieldCentres[
                1, Currentiteration, tdim1, tdim2] != 0:
                fieldlistdim1.append(FieldCentres[0, Currentiteration, tdim1, tdim2])
                fieldlistdim2.append(FieldCentres[1, Currentiteration, tdim1, tdim2])
        for fieldcell in range(len(fieldlistdim1) - 1):
            totaldistance += np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2)
            count += 1

    # Field distance with closest neighbours in dim1
    for tdim2 in range(border, len(Wpt[0, 0, :, 0, 0]) - border):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim1 in range(border, len(Wpt[0, :, 0, 0, 0]) - border):
            if FieldCentres[0, Currentiteration, tdim1, tdim2] != 0 and FieldCentres[
                1, Currentiteration, tdim1, tdim2] != 0:
                fieldlistdim1.append(FieldCentres[0, Currentiteration, tdim1, tdim2])
                fieldlistdim2.append(FieldCentres[1, Currentiteration, tdim1, tdim2])
        for fieldcell in range(len(fieldlistdim1) - 1):
            totaldistance += np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2)
            count += 1

    meanseparation = totaldistance / count
    FieldSeparation[Currentiteration] = meanseparation


def field_size(border):
    totaldiameter = 0
    count = 0
    for tdim1 in range(border, len(Wpt[0, :, 0, 0, 0]) - border):
        for tdim2 in range(border, len(Wpt[0, 0, :, 0, 0]) - border):

            area = 0
            # Scanning in dim1
            for rdim2 in range(len(Wpt[0, 0, 0, 0, :])):
                width = 0
                rdim1 = 0
                weight = 0
                while weight == 0 and rdim1 < len(Wpt[0, 0, 0, :, 0]):
                    weight = Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]
                    rdim1 += 1
                min = rdim1 - 1
                if weight != 0:
                    rdim1 = len(Wpt[0, 0, 0, :, 0]) - 1
                    weight = 0
                    while weight == 0:
                        weight = Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]
                        rdim1 -= 1
                    max = rdim1 + 1
                    width = max - min
                area += width

            # Scanning in dim2
            for rdim1 in range(len(Wpt[0, 0, 0, :, 0])):
                width = 0
                rdim2 = 0
                weight = 0
                while weight == 0 and rdim2 < len(Wpt[0, 0, 0, 0, :]):
                    weight = Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]
                    rdim2 += 1
                min = rdim2 - 1
                if weight != 0:
                    rdim2 = len(Wpt[0, 0, 0, 0, :]) - 1
                    weight = 0
                    while weight == 0:
                        weight = Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]
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
    FieldSize[Currentiteration] = meandiameter


def systems_match(border):
    totaldistance = 0
    count = 0

    for tdim1 in range(border, len(Wpt[0, :, 0, 0, 0]) - border):
        for tdim2 in range(border, len(Wpt[0, 0, :, 0, 0]) - border):
            if FieldCentres[0, Currentiteration, tdim1, tdim2] != 0 and FieldCentres[
                1, Currentiteration, tdim1, tdim2] != 0:
                totaldistance += np.sqrt((FieldCentres[0, Currentiteration, tdim1, tdim2] - xFieldCentres[
                    0, Weightiteration, tdim1, tdim2]) ** 2 + (
                                             FieldCentres[1, Currentiteration, tdim1, tdim2] - xFieldCentres[
                                                 1, Weightiteration, tdim1, tdim2]) ** 2)
                count += 1

    meandistance = totaldistance / count
    SystemsMatch[Currentiteration] = meandistance


def savedata(JobID, Timecompression):
    np.save('../../RetinotopicMapsData/%s/FieldSizeEB' % ('{0:04}'.format(JobID)), FieldSize)
    np.save('../../RetinotopicMapsData/%s/FieldSeparationEB' % ('{0:04}'.format(JobID)), FieldSeparation)
    np.save('../../RetinotopicMapsData/%s/SystemsMatchEB' % ('{0:04}'.format(JobID)), SystemsMatch)
    np.save('../../RetinotopicMapsData/%s/SecondaryTREB' % ('{0:04}'.format(JobID)), TRin * Timecompression)
