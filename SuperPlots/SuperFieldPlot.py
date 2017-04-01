import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

start = time.time()

##################### IMPORT DATA ########################

JobID = int(input('JobID: '))
print('Loading Data...')
Weightmatrix = np.load('../../RetinotopicMapsData/%s/Weightmatrix.npy' % ('{0:04}'.format(JobID)))
Fieldcentres = np.load('../../RetinotopicMapsData/%s/FieldCentres.npy' % ('{0:04}'.format(JobID)))
TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

######################## PLOT OPTIONS #####################

Iterations = [1, 10, 20, 50, 100, 500]


####################### PLOT ##########################
fig = plt.figure()
figs = fig.add_subplot(111, frameon=False)
figs.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
figs.set_ylabel('Retinal Cell Number (Dimension 2)')
figs.set_xlabel('Retinal Cell Number (Dimension 1)')


def fieldplot(plotn):
    i = Iterations[plotn]

    ax = fig.add_subplot(2, 3, plotn + 1)
    ax.set_xlim(0, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    ax.set_ylim(0, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
    ax.set_title('%d iterations' % i)

    for tdim1 in range(len(Fieldcentres[0, i // TRin, :, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim2 in range(len(Fieldcentres[0, i // TRin, 0, :])):
            if Fieldcentres[0, i // TRin, tdim1, tdim2] != 0 and Fieldcentres[1, i // TRin, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, i // TRin, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, i // TRin, tdim1, tdim2])
        ax.plot(fieldlistdim1, fieldlistdim2, c='k', lw='0.5')

    for tdim2 in range(len(Fieldcentres[0, i // TRin, 0, :])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim1 in range(len(Fieldcentres[0, i // TRin, :, 0])):
            if Fieldcentres[0, i // TRin, tdim1, tdim2] != 0 and Fieldcentres[1, i // TRin, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, i // TRin, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, i // TRin, tdim1, tdim2])
        ax.plot(fieldlistdim1, fieldlistdim2, c='k', lw='0.5')

    # for tdim1 in range(5, len(Fieldcentres[0, i // TRin, :, 0]) - 5):
    #     fieldlistdim1 = []
    #     fieldlistdim2 = []
    #     for tdim2 in range(5, len(Fieldcentres[0, i // TRin, 0, :]) - 5):
    #         if Fieldcentres[0, i // TRin, tdim1, tdim2] != 0 and Fieldcentres[1, i // TRin, tdim1, tdim2] != 0:
    #             fieldlistdim1.append(Fieldcentres[0, i // TRin, tdim1, tdim2])
    #             fieldlistdim2.append(Fieldcentres[1, i // TRin, tdim1, tdim2])
    #     ax.plot(fieldlistdim1, fieldlistdim2, c='k', lw='0.1')
    #
    # for tdim2 in range(5, len(Fieldcentres[0, i // TRin, 0, :]) - 5):
    #     fieldlistdim1 = []
    #     fieldlistdim2 = []
    #     for tdim1 in range(5, len(Fieldcentres[0, i // TRin, :, 0]) - 5):
    #         if Fieldcentres[0, i // TRin, tdim1, tdim2] != 0 and Fieldcentres[1, i // TRin, tdim1, tdim2] != 0:
    #             fieldlistdim1.append(Fieldcentres[0, i // TRin, tdim1, tdim2])
    #             fieldlistdim2.append(Fieldcentres[1, i // TRin, tdim1, tdim2])
    #     ax.plot(fieldlistdim1, fieldlistdim2, c='k', lw='0.1')

    if (plotn+1) == 1:
        ax.set_xticklabels([])
    if (plotn+1) == 2 or (plotn+1) == 3:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if (plotn+1) == 5 or (plotn+1) == 6:
        ax.set_yticklabels([])

for plotn in range(len(Iterations)):
    fieldplot(plotn)

plt.show()
