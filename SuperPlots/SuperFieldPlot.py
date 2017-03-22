import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

##################### IMPORT DATA ########################

JobID = int(input('JobID: '))

Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
Fieldcentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(JobID)))
TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

######################## PLOT OPTIONS #####################

Iterations = [0, 1000, 2000, 3000, 4000, 5000]


####################### PLOT ##########################

def fieldplot(plotn):
    i = Iterations[plotn] // TRin

    plt.subplot(2, 3, plotn + 1)
    plt.xlim(1, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    plt.ylim(1, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
    plt.xlabel('Retinal Cell Number (Dimension 1)')
    plt.ylabel('Retinal Cell Number (Dimension 2)')
    plt.title('%d iterations' % (i * TRin))

    for tdim1 in range(len(Fieldcentres[0, i, :, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim2 in range(len(Fieldcentres[0, i, 0, :])):
            if Fieldcentres[0, i, tdim1, tdim2] != 0 and Fieldcentres[1, i, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, i, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, i, tdim1, tdim2])

        plt.plot(fieldlistdim1, fieldlistdim2, c='0.5')

    for tdim2 in range(len(Fieldcentres[0, i, 0, :])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim1 in range(len(Fieldcentres[0, i, :, 0])):
            if Fieldcentres[0, i, tdim1, tdim2] != 0 and Fieldcentres[1, i, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, i, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, i, tdim1, tdim2])

        plt.plot(fieldlistdim1, fieldlistdim2, c='0.5')

    for tdim1 in range(5, len(Fieldcentres[0, i, :, 0]) - 5):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim2 in range(5, len(Fieldcentres[0, i, 0, :]) - 5):
            if Fieldcentres[0, i, tdim1, tdim2] != 0 and Fieldcentres[1, i, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, i, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, i, tdim1, tdim2])

        plt.plot(fieldlistdim1, fieldlistdim2, c='k')

    for tdim2 in range(5, len(Fieldcentres[0, i, 0, :]) - 5):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim1 in range(5, len(Fieldcentres[0, i, :, 0]) - 5):
            if Fieldcentres[0, i, tdim1, tdim2] != 0 and Fieldcentres[1, i, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, i, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, i, tdim1, tdim2])

        plt.plot(fieldlistdim1, fieldlistdim2, c='k')


for plotn in range(len(Iterations)):
    fieldplot(plotn)

plt.show()
