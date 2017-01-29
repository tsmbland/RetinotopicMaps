import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
start = time.time()


##################### IMPORT DATA ########################

Weightmatrix = np.load('Weightmatrix.npy')
Fieldcentres = np.load('Fieldcentres.npy')


####################### PLOT ##########################

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(1, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
ax.set_ylim(1, len(Weightmatrix[0, 0, 0, 0, :]) - 2)
ax.set_xlabel('Retinal Cell Number (Dimension 1)')
ax.set_ylabel('Retinal Cell Number (Dimension 2)')
plt.subplots_adjust(left=0.25, bottom=0.25)
axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
sframe = Slider(axframe, 'Iteration', 0, len(Fieldcentres[0, :, 0, 0]) - 1, valinit=0, valfmt='%d')


def fieldplot(i):
    for tdim1 in range(len(Fieldcentres[0, i, :, 0])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim2 in range(len(Fieldcentres[0, i, 0, :])):
            if Fieldcentres[0, i, tdim1, tdim2] != 0 and Fieldcentres[1, i, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, i, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, i, tdim1, tdim2])

        ax.plot(fieldlistdim1, fieldlistdim2, c='k')

    for tdim2 in range(len(Fieldcentres[0, i, 0, :])):
        fieldlistdim1 = []
        fieldlistdim2 = []
        for tdim1 in range(len(Fieldcentres[0, i, :, 0])):
            if Fieldcentres[0, i, tdim1, tdim2] != 0 and Fieldcentres[1, i, tdim1, tdim2] != 0:
                fieldlistdim1.append(Fieldcentres[0, i, tdim1, tdim2])
                fieldlistdim2.append(Fieldcentres[1, i, tdim1, tdim2])

        ax.plot(fieldlistdim1, fieldlistdim2, c='k')


def update(val):
    it = np.floor(sframe.val)
    ax.clear()
    fieldplot(it)
    ax.set_xlim(1, len(Weightmatrix[0, 0, 0, :, 0]) - 2)
    ax.set_ylim(1, len(Weightmatrix[0, 0, 0, 0, :]) - 2)


fieldplot(0)

###################### END ########################
params = {'font.size': '10'}
plt.rcParams.update(params)
sframe.on_changed(update)
plt.show()
