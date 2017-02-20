import matplotlib.pyplot as plt
import numpy as np

##################### IMPORT DATA #####################

Fieldsize = np.load('../Temporary Data/Fieldsize.npy')
Fieldseparation = np.load('../Temporary Data/Fieldseparation.npy')
Systemsmatch = np.load('../Temporary Data/Systemsmatch.npy')

###################### OPTIONS #######################

TRin = 10  # Temporal resolution of input files

######################## PLOTS #######################
plt.subplot(1, 3, 1)
plt.title('Receptive Field Separation')
plt.plot(range(TRin, len(Fieldseparation) * TRin, TRin), Fieldseparation[1:])
plt.ylabel('Mean Receptive Field Separation')
plt.xlabel('Time')
plt.xlim(0, len(Fieldseparation) * TRin)

plt.subplot(1, 3, 2)
plt.title('Receptive Field Size')
plt.plot(range(TRin, len(Fieldsize) * TRin, TRin), Fieldsize[1:])
plt.ylabel('Mean Receptive Field Diameter')
plt.xlabel('Time')
plt.xlim(0, len(Fieldsize) * TRin)

plt.subplot(1, 3, 3)
plt.title('Systems Match')
plt.plot(range(TRin, len(Systemsmatch) * TRin, TRin), Systemsmatch[1:])
plt.ylabel('Mean Distance Between Field Centre and Expected Field Centre')
plt.xlabel('Time')
plt.xlim(0, len(Systemsmatch) * TRin)

###################### END ########################
params = {'font.size': '10'}
plt.rcParams.update(params)
plt.show()
