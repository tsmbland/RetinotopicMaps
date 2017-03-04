import matplotlib.pyplot as plt
import numpy as np

##################### IMPORT DATA #####################

JobID = int(input('JobID: '))

Meanchange = np.load('../../RetinotopicMapsData/%s/MeanChange.npy' % ('{0:04}'.format(JobID)))

###################### OPTIONS #######################

TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(JobID)))

######################## PLOTS #######################

plt.title('Mean Change')
plt.plot(range(TRin, len(Meanchange) * TRin, TRin), Meanchange[1:])
plt.ylabel('Mean Change')
plt.xlabel('Time')
plt.xlim(0, len(Meanchange) * TRin)

plt.show()
