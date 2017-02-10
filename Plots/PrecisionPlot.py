import matplotlib.pyplot as plt
import numpy as np

##################### IMPORT DATA #####################

Fieldsize = np.load('../Temporary Data/Fieldsize.npy')
Fieldseparation = np.load('../Temporary Data/Fieldseparation.npy')


######################## PLOTS #######################
plt.subplot(1, 2, 1)
plt.title('Receptive Field Separation')
plt.plot(Fieldseparation)
plt.ylabel('Mean Receptive Field Separation')
plt.xlabel('Time')

plt.subplot(1, 2, 2)
plt.title('Receptive Field Size')
plt.plot(Fieldsize)
plt.ylabel('Mean Receptive Field Diameter')
plt.xlabel('Time')

###################### END ########################
params = {'font.size': '10'}
plt.rcParams.update(params)
plt.show()
