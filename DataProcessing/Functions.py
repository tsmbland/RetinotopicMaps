import numpy as np


####################### FUNCTIONS #####################

def importdata(jobid, timecompression):
    global Wpt, xFieldCentres, FieldCentres, FieldSeparation, FieldSizes, SystemsMatch, MeanChange, TRin, Currentiteration, Weightiteration
    Wpt = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(jobid)))
    xFieldCentres = np.load('../../RetinotopicMapsData/%s/xFieldCentres.npy' % ('{0:04}'.format(jobid)))
    TRin = np.load('../../RetinotopicMapsData/%s/PrimaryTR.npy' % ('{0:04}'.format(jobid)))

    FieldCentres = np.zeros(
        [2, int(len(Wpt[:, 0, 0, 0, 0]) / timecompression), len(Wpt[0, :, 0, 0, 0]), len(Wpt[0, 0, :, 0, 0])])
    FieldSizes = np.zeros(
        [int(len(Wpt[:, 0, 0, 0, 0]) / timecompression), len(Wpt[0, :, 0, 0, 0]), len(Wpt[0, 0, :, 0, 0])])

    Currentiteration = -1
    Weightiteration = -1


def updatetimepoint(timecompression):
    global Currentiteration, Weightiteration
    Currentiteration += 1
    Weightiteration += timecompression


def UpdateFieldCentres():
    synapses = np.array(np.nonzero(Wpt[Weightiteration, :, :, :, :]))

    totaldim1 = np.zeros([len(Wpt[0, :, 0, 0, 0]), len(Wpt[0, 0, :, 0, 0])])
    totaldim2 = np.zeros([len(Wpt[0, :, 0, 0, 0]), len(Wpt[0, 0, :, 0, 0])])
    weightsumdim1 = np.zeros([len(Wpt[0, :, 0, 0, 0]), len(Wpt[0, 0, :, 0, 0])])
    weightsumdim2 = np.zeros([len(Wpt[0, :, 0, 0, 0]), len(Wpt[0, 0, :, 0, 0])])

    for synapse in range(int(len(synapses[0, :]))):
        tdim1 = synapses[0, synapse]
        tdim2 = synapses[1, synapse]
        rdim1 = synapses[2, synapse]
        rdim2 = synapses[3, synapse]
        weightsumdim1[tdim1, tdim2] += Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]
        weightsumdim2[tdim1, tdim2] += Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]
        totaldim1[tdim1, tdim2] += rdim1 * Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]
        totaldim2[tdim1, tdim2] += rdim2 * Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]

    FieldCentres[0, Currentiteration, :, :] = totaldim1 / weightsumdim1
    FieldCentres[1, Currentiteration, :, :] = totaldim2 / weightsumdim2
    FieldCentres[:, Currentiteration, :, :] = np.nan_to_num(FieldCentres[:, Currentiteration, :, :])



def UpdateFieldSizes():
    for tdim1 in range(len(Wpt[0, :, 0, 0, 0])):
        for tdim2 in range(len(Wpt[0, 0, :, 0, 0])):

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
            FieldSizes[Currentiteration, tdim1, tdim2] = diameter


def savedata(JobID, Timecompression):
    #np.save('../../RetinotopicMapsData/%s/FieldCentres' % ('{0:04}'.format(JobID)), FieldCentres)
    np.save('../../RetinotopicMapsData/%s/FieldSizes' % ('{0:04}'.format(JobID)), FieldSizes)
    np.save('../../RetinotopicMapsData/%s/SecondaryTR' % ('{0:04}'.format(JobID)), TRin * Timecompression)
