import numpy as np


####################### FUNCTIONS #####################

def importdata(jobid):
    global xFieldCentres, TRin, FieldCentres, FieldSizes
    global FieldSeparation, FieldSeparationStdev, FieldSize, SystemsMatch, FieldSeparationEB, FieldSeparationStdevEB, FieldSizeEB, SystemsMatchEB
    global Currentiteration, Weightiteration

    xFieldCentres = np.load('../../RetinotopicMapsData/%s/xFieldCentres.npy' % ('{0:04}'.format(jobid)))
    TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(jobid)))
    FieldCentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(jobid)))
    FieldSizes = np.load('../../RetinotopicMapsData/%s/FieldSizes.npy' % ('{0:04}'.format(jobid)))

    FieldSeparation = np.zeros(int(len(FieldCentres[0, :, 0, 0])))
    FieldSeparationStdev = np.zeros(int(len(FieldCentres[0, :, 0, 0])))
    FieldSize = np.zeros(int(len(FieldCentres[0, :, 0, 0])))
    SystemsMatch = np.zeros(int(len(FieldCentres[0, :, 0, 0])))

    FieldSeparationEB = np.zeros(int(len(FieldCentres[0, :, 0, 0])))
    FieldSeparationStdevEB = np.zeros(int(len(FieldCentres[0, :, 0, 0])))
    FieldSizeEB = np.zeros(int(len(FieldCentres[0, :, 0, 0])))
    SystemsMatchEB = np.zeros(int(len(FieldCentres[0, :, 0, 0])))

    Currentiteration = -1
    Weightiteration = -1


def updatetimepoint():
    global Currentiteration, Weightiteration
    Currentiteration += 1
    Weightiteration += 1


def MeanFieldSeparation():
    fieldseparations = []

    # Field distance with closest neighbours in dim2
    for tdim1 in range(len(FieldCentres[0, 0, :, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim2 in range(len(FieldCentres[0, 0, 0, :])):
            if FieldCentres[0, Currentiteration, tdim1, tdim2] != 0 and FieldCentres[
                1, Currentiteration, tdim1, tdim2] != 0:
                fieldlistdim1.append(FieldCentres[0, Currentiteration, tdim1, tdim2])
                fieldlistdim2.append(FieldCentres[1, Currentiteration, tdim1, tdim2])
        for fieldcell in range(len(fieldlistdim1) - 1):
            fieldseparations.append(np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2))

    # Field distance with closest neighbours in dim1
    for tdim2 in range(len(FieldCentres[0, 0, 0, :])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim1 in range(len(FieldCentres[0, 0, :, 0])):
            if FieldCentres[0, Currentiteration, tdim1, tdim2] != 0 and FieldCentres[
                1, Currentiteration, tdim1, tdim2] != 0:
                fieldlistdim1.append(FieldCentres[0, Currentiteration, tdim1, tdim2])
                fieldlistdim2.append(FieldCentres[1, Currentiteration, tdim1, tdim2])
        for fieldcell in range(len(fieldlistdim1) - 1):
            fieldseparations.append(np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2))

    meanseparation = np.mean(fieldseparations)
    standarddeviation = np.std(fieldseparations)
    FieldSeparation[Currentiteration] = meanseparation
    FieldSeparationStdev[Currentiteration] = standarddeviation


def MeanFieldSize():
    totalsize = 0
    count = 0
    for tdim1 in range(len(FieldCentres[0, 0, :, 0])):
        for tdim2 in range(len(FieldCentres[0, 0, 0, :])):
            if FieldCentres[0, Currentiteration, tdim1, tdim2] != 0 and FieldCentres[
                1, Currentiteration, tdim1, tdim2] != 0:
                totalsize += FieldSizes[Currentiteration, tdim1, tdim2]
                count += 1
    FieldSize[Currentiteration] = totalsize / count


def MeanSystemsMatch():
    separation = []

    for tdim1 in range(len(FieldCentres[0, 0, :, 0])):
        for tdim2 in range(len(FieldCentres[0, 0, 0, :])):
            if FieldCentres[0, Currentiteration, tdim1, tdim2] != 0 and FieldCentres[
                1, Currentiteration, tdim1, tdim2] != 0:
                separation.append(np.sqrt((FieldCentres[0, Currentiteration, tdim1, tdim2] - xFieldCentres[
                    0, Weightiteration, tdim1, tdim2]) ** 2 + (
                                              FieldCentres[1, Currentiteration, tdim1, tdim2] - xFieldCentres[
                                                  1, Weightiteration, tdim1, tdim2]) ** 2))

    meandistance = np.mean(separation)
    SystemsMatch[Currentiteration] = meandistance


def MeanFieldSeparationEB(border):
    fieldseparations = []

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
            fieldseparations.append(np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2))

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
            fieldseparations.append(np.sqrt((fieldlistdim1[fieldcell] - fieldlistdim1[fieldcell + 1]) ** 2 + (
                fieldlistdim2[fieldcell] - fieldlistdim2[fieldcell + 1]) ** 2))

    meanseparation = np.mean(fieldseparations)
    standarddeviation = np.std(fieldseparations)
    FieldSeparationEB[Currentiteration] = meanseparation
    FieldSeparationStdevEB[Currentiteration] = standarddeviation


def MeanFieldSizeEB(border):
    totalsize = 0
    count = 0
    for tdim1 in range(border, len(FieldCentres[0, 0, :, 0]) - border):
        for tdim2 in range(border, len(FieldCentres[0, 0, 0, :]) - border):
            if FieldCentres[0, Currentiteration, tdim1, tdim2] != 0 and FieldCentres[
                1, Currentiteration, tdim1, tdim2] != 0:
                totalsize += FieldSizes[Currentiteration, tdim1, tdim2]
                count += 1
    FieldSizeEB[Currentiteration] = totalsize / count


def MeanSystemsMatchEB(border):
    separation = []

    for tdim1 in range(border, len(FieldCentres[0, 0, :, 0]) - border):
        for tdim2 in range(border, len(FieldCentres[0, 0, 0, :]) - border):
            if FieldCentres[0, Currentiteration, tdim1, tdim2] != 0 and FieldCentres[
                1, Currentiteration, tdim1, tdim2] != 0:
                separation.append(np.sqrt((FieldCentres[0, Currentiteration, tdim1, tdim2] - xFieldCentres[
                    0, Weightiteration, tdim1, tdim2]) ** 2 + (
                                              FieldCentres[1, Currentiteration, tdim1, tdim2] - xFieldCentres[
                                                  1, Weightiteration, tdim1, tdim2]) ** 2))

    meandistance = np.mean(separation)
    SystemsMatchEB[Currentiteration] = meandistance



def savedata(JobID):
    np.save('../../RetinotopicMapsData/%s/FieldSize' % ('{0:04}'.format(JobID)), FieldSize)
    np.save('../../RetinotopicMapsData/%s/FieldSeparation' % ('{0:04}'.format(JobID)), FieldSeparation)
    np.save('../../RetinotopicMapsData/%s/FieldSeparationStdev' % ('{0:04}'.format(JobID)), FieldSeparationStdev)
    np.save('../../RetinotopicMapsData/%s/SystemsMatch' % ('{0:04}'.format(JobID)), SystemsMatch)

    np.save('../../RetinotopicMapsData/%s/FieldSizeEB' % ('{0:04}'.format(JobID)), FieldSizeEB)
    np.save('../../RetinotopicMapsData/%s/FieldSeparationEB' % ('{0:04}'.format(JobID)), FieldSeparationEB)
    np.save('../../RetinotopicMapsData/%s/FieldSeparationStdevEB' % ('{0:04}'.format(JobID)), FieldSeparationStdevEB)
    np.save('../../RetinotopicMapsData/%s/SystemsMatchEB' % ('{0:04}'.format(JobID)), SystemsMatchEB)
