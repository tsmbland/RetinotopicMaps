import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['savefig.dpi'] = 600

Dimensions = [10, 20, 30, 40, 50]

HybridSpeed = [0.1526, 0.9517, 2.7, 5.617, 10.06]
OriginalSpeed = [0.1648, 2.478, 11.9, 36, 85.73]

plt.plot(Dimensions, OriginalSpeed, 'r', label='Original (2006) Model')
plt.plot(Dimensions, OriginalSpeed, 'ro')

plt.plot(Dimensions, HybridSpeed, 'b', label='New Model')
plt.plot(Dimensions, HybridSpeed, 'bo')


plt.xlabel('Cells per dimension')
plt.ylabel('Time per iteration (s)')

plt.legend(loc=2)
plt.show()
