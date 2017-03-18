import numpy as np


####################### FUNCTIONS #####################

def importdata(jobid, timecompression):
    global xFieldCentres, FieldCentres, FieldSizes, FieldSeparation, FieldSize, SystemsMatch, MeanChange, TRin, Currentiteration, Weightiteration
    xFieldCentres = np.load('../../RetinotopicMapsData/%s/xFieldCentres.npy' % ('{0:04}'.format(jobid)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(jobid)))
    FieldCentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(jobid)))
    FieldSizes = np.load('../../RetinotopicMapsData/%s/FieldSizes.npy' % ('{0:04}'.format(jobid)))

    FieldSeparation = np.zeros(int(len(FieldCentres[0, :, 0, 0]) / timecompression))
    FieldSize = np.zeros(int(len(FieldCentres[0, :, 0, 0]) / timecompression))
    SystemsMatch = np.zeros(int(len(FieldCentres[0, :, 0, 0]) / timecompression))

    Currentiteration = -1
    Weightiteration = -1


def updatetimepoint(timecompression):
    global Currentiteration, Weightiteration
    Currentiteration += 1
    Weightiteration += timecompression


def MeanFieldSeparation(border):
    totaldistance = 0
    count = 0

    # Field distance with closest neighbours in dim2
    for tdim1 in range(border, len(FieldCentres[0, 0, :, 0]) - border):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim2 in range(border, len(FieldCentres[0, 0, 0, :]) - border):
            if FieldCentres[0, Currentiteration, tdim1, tdim2] != 0 and FieldCentres[
                1, Currentiteration, tdim1, tdim2] != 0:
                fieldlistdim1.append(FieldCentres[0, Currentiteration, tdim1, tdim2])
                fieldlistdim2.append(FieldCentres[1, Currentiteration, tdim1, tdim2])
        for fieldcell in range(len(fieldlistdim1) - 1):
            totaldistance += np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2)
            count += 1

    # Field distance with closest neighbours in dim1
    for tdim2 in range(border, len(FieldCentres[0, 0, 0, :]) - border):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim1 in range(border, len(FieldCentres[0, 0, :, 0]) - border):
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


def MeanFieldSize(border):
    FieldSize[Currentiteration] = np.mean(
        np.mean(FieldSizes[Currentiteration, (border + 1):-(1 + border), (border + 1):- (1 + border)]))


def MeanSystemsMatch(border):
    totaldistance = 0
    count = 0

    for tdim1 in range(border, len(FieldCentres[0, 0, :, 0]) - border):
        for tdim2 in range(border, len(FieldCentres[0, 0, 0, :]) - border):
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
    np.save('../../RetinotopicMapsData/%s/FieldSize' % ('{0:04}'.format(JobID)), FieldSize)
    np.save('../../RetinotopicMapsData/%s/FieldSeparation' % ('{0:04}'.format(JobID)), FieldSeparation)
    np.save('../../RetinotopicMapsData/%s/SystemsMatch' % ('{0:04}'.format(JobID)), SystemsMatch)


def savedataEB(JobID, Timecompression):
    np.save('../../RetinotopicMapsData/%s/FieldSizeEB' % ('{0:04}'.format(JobID)), FieldSize)
    np.save('../../RetinotopicMapsData/%s/FieldSeparationEB' % ('{0:04}'.format(JobID)), FieldSeparation)
    np.save('../../RetinotopicMapsData/%s/SystemsMatchEB' % ('{0:04}'.format(JobID)), SystemsMatch)