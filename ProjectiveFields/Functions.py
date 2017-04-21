import numpy as np


####################### FUNCTIONS #####################

def importdata(jobid, timecompression):
    global Wpt, FieldCentres, FieldSeparation, FieldSizes, SystemsMatch, MeanChange, TRin, Currentiteration, Weightiteration
    Wpt = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(jobid)))
    TRin = np.load('../../RetinotopicMapsData/%s/PrimaryTR.npy' % ('{0:04}'.format(jobid)))

    FieldCentres = np.zeros(
        [2, int(len(Wpt[:, 0, 0, 0, 0]) / timecompression), len(Wpt[0, 0, 0, :, 0]), len(Wpt[0, 0, 0, 0, :])])
    FieldSizes = np.zeros(
        [int(len(Wpt[:, 0, 0, 0, 0]) / timecompression), len(Wpt[0, 0, 0, :, 0]), len(Wpt[0, 0, 0, 0, :])])

    Currentiteration = -1
    Weightiteration = -1


def updatetimepoint(timecompression):
    global Currentiteration, Weightiteration
    Currentiteration += 1
    Weightiteration += timecompression


def UpdateFieldCentres():
    synapses = np.array(np.nonzero(Wpt[Weightiteration, :, :, :, :]))

    totaldim1 = np.zeros([len(Wpt[0, 0, 0, :, 0]), len(Wpt[0, 0, 0, 0, :])])
    totaldim2 = np.zeros([len(Wpt[0, 0, 0, :, 0]), len(Wpt[0, 0, 0, 0, :])])
    weightsumdim1 = np.zeros([len(Wpt[0, 0, 0, :, 0]), len(Wpt[0, 0, 0, 0, :])])
    weightsumdim2 = np.zeros([len(Wpt[0, 0, 0, :, 0]), len(Wpt[0, 0, 0, 0, :])])

    for synapse in range(int(len(synapses[0, :]))):
        tdim1 = synapses[0, synapse]
        tdim2 = synapses[1, synapse]
        rdim1 = synapses[2, synapse]
        rdim2 = synapses[3, synapse]
        weightsumdim1[rdim1, rdim2] += Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]
        weightsumdim2[rdim1, rdim2] += Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]
        totaldim1[rdim1, rdim2] += tdim1 * Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]
        totaldim2[rdim1, rdim2] += tdim2 * Wpt[Weightiteration, tdim1, tdim2, rdim1, rdim2]

    FieldCentres[0, Currentiteration, :, :] = totaldim1 / weightsumdim1
    FieldCentres[1, Currentiteration, :, :] = totaldim2 / weightsumdim2
    FieldCentres[:, Currentiteration, :, :] = np.nan_to_num(FieldCentres[:, Currentiteration, :, :])


def UpdateFieldSizes():
    for rdim1 in range(len(Wpt[0, 0, 0, :, 0])):
        for rdim2 in range(len(Wpt[0, 0, 0, 0, :])):
            area = 0
            synapses = np.array(np.nonzero(Wpt[Weightiteration, :, :, rdim1, rdim2]))
            for dim1 in np.unique(synapses[0, :]):
                area += max(synapses[1, synapses[0, :] == dim1]) - min(synapses[1, synapses[0, :] == dim1])

            for dim2 in np.unique(synapses[1, :]):
                area += max(synapses[0, synapses[1, :] == dim2]) - min(synapses[0, synapses[1, :] == dim2])

            diameter = 2 * np.sqrt(area / (2 * np.pi))
            FieldSizes[Currentiteration, rdim1, rdim2] = diameter


def savedata(JobID, Timecompression):
    np.save('../../RetinotopicMapsData/%s/ReverseFieldCentres' % ('{0:04}'.format(JobID)), FieldCentres)
    np.save('../../RetinotopicMapsData/%s/ReverseFieldSizes' % ('{0:04}'.format(JobID)), FieldSizes)
    np.save('../../RetinotopicMapsData/%s/ReverseSecondaryTR' % ('{0:04}'.format(JobID)), TRin * Timecompression)
