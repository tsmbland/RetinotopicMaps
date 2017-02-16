import matplotlib.pyplot as plt
import numpy as np

##################### IMPORT DATA #####################

Fieldsize = np.load('../Temporary Data/Fieldsize.npy')
Fieldseparation = np.load('../Temporary Data/Fieldseparation.npy')
Systemsmatch = np.load('../Temporary Data/Systemsmatch.npy')

###################### OPTIONS #######################

TRin = 5  # Temporal resolution of input files

######################## PLOTS #######################
plt.subplot(1, 3, 1)
plt.title('Receptive Field Separation')
plt.plot(range(0, len(Fieldseparation) * TRin, TRin), Fieldseparation)
plt.ylabel('Mean Receptive Field Separation')
plt.xlabel('Time')

plt.subplot(1, 3, 2)
plt.title('Receptive Field Size')
plt.plot(range(0, len(Fieldsize) * TRin, TRin), Fieldsize)
plt.ylabel('Mean Receptive Field Diameter')
plt.xlabel('Time')

plt.subplot(1, 3, 3)
plt.title('Systems Match')
plt.plot(range(0, len(Fieldsize) * TRin, TRin), Systemsmatch)
plt.ylabel('Mean Distance Between Field Centre and Expected Field Centre')
plt.xlabel('Time')

###################### END ########################
params = {'font.size': '10'}
plt.rcParams.update(params)
plt.show()
