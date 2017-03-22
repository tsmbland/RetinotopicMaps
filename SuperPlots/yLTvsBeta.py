import numpy as np
import matplotlib.pyplot as plt

#################### JOBS ###################

yLT = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
beta = [0, 0.001, 0.01, 0.1, 0.2, 0.5, 1.]

JobID = np.zeros([len(yLT), len(beta)], dtype=np.int8)
JobID[0, :] = [0, 126, 99, 108, 117, 135, 0]
JobID[1, :] = [0, 125, 98, 107, 116, 134, 0]
JobID[2, :] = [0, 124, 97, 106, 115, 133, 0]
JobID[3, :] = [0, 123, 96, 105, 114, 132, 0]
JobID[4, :] = [0, 122, 95, 104, 113, 131, 0]
JobID[5, :] = [0, 121, 94, 103, 112, 130, 0]
JobID[6, :] = [0, 120, 93, 102, 111, 129, 0]
JobID[7, :] = [0, 119, 92, 101, 110, 128, 0]
JobID[8, :] = [0, 85, 86, 87, 88, 89, 90]
JobID[9, :] = [0, 118, 91, 100, 109, 127, 0]
JobID[10, :] = [0, 79, 80, 81, 82, 83, 84]

################# DATA MATRICES ###############

Sep = np.zeros([len(yLT), len(beta)])
Size = np.zeros([len(yLT), len(beta)])
SM = np.zeros([len(yLT), len(beta)])

SepEB = np.zeros([len(yLT), len(beta)])
SizeEB = np.zeros([len(yLT), len(beta)])
SMEB = np.zeros([len(yLT), len(beta)])

SepStab = np.zeros([len(yLT), len(beta)])
SizeStab = np.zeros([len(yLT), len(beta)])
SMStab = np.zeros([len(yLT), len(beta)])

SepStabEB = np.zeros([len(yLT), len(beta)])
SizeStabEB = np.zeros([len(yLT), len(beta)])
SMStabEB = np.zeros([len(yLT), len(beta)])

################### IMPORT DATA ################

for row in range(len(yLT)):
    for column in range(2, 5):
        ID = JobID[row, column]
        if ID != 0:
            # Import Data
            Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSize.npy' % ('{0:04}'.format(ID)))
            Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparation.npy' % ('{0:04}'.format(ID)))
            Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatch.npy' % ('{0:04}'.format(ID)))

            FieldsizeEB = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(ID)))
            FieldseparationEB = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(ID)))
            SystemsmatchEB = np.load('../../RetinotopicMapsData/%s/SystemsMatchEB.npy' % ('{0:04}'.format(ID)))

            Fieldsizechange = np.load('../../RetinotopicMapsData/%s/FieldSizeChange.npy' % ('{0:04}'.format(ID)))
            Fieldseparationchange = np.load(
                '../../RetinotopicMapsData/%s/FieldSeparationChange.npy' % ('{0:04}'.format(ID)))
            Systemsmatchchange = np.load('../../RetinotopicMapsData/%s/SystemsMatchChange.npy' % ('{0:04}'.format(ID)))

            FieldsizechangeEB = np.load('../../RetinotopicMapsData/%s/FieldSizeChangeEB.npy' % ('{0:04}'.format(ID)))
            FieldseparationchangeEB = np.load(
                '../../RetinotopicMapsData/%s/FieldSeparationChangeEB.npy' % ('{0:04}'.format(ID)))
            SystemsmatchchangeEB = np.load(
                '../../RetinotopicMapsData/%s/SystemsMatchChangeEB.npy' % ('{0:04}'.format(ID)))

            # Find stable time
            sizechangethresh = 0.005
            smthresh = 0.01
            fieldsizechange = 1
            systemsmatchchange = 1
            iteration = 0
            while (fieldsizechange > sizechangethresh or fieldsizechange < -sizechangethresh or
                           systemsmatchchange > smthresh or systemsmatchchange < -smthresh) and iteration < len(
                Fieldsizechange):
                fieldsizechange = Fieldsizechange[iteration]
                systemsmatchchange = Systemsmatchchange[iteration]
                iteration += 1

            # Fill matrices
            Sep[row, column] = Fieldseparation[-1]
            Size[row, column] = Fieldsize[-1]
            SM[row, column] = Systemsmatch[-1]

            SepEB[row, column] = FieldseparationEB[-1]
            SizeEB[row, column] = FieldsizeEB[-1]
            SMEB[row, column] = SystemsmatchEB[-1]

            SepStab[row, column] = Fieldseparation[iteration]
            SizeStab[row, column] = Fieldsize[iteration]
            SMStab[row, column] = Systemsmatch[iteration]

            SepStabEB[row, column] = FieldseparationEB[iteration]
            SizeStabEB[row, column] = FieldsizeEB[iteration]
            SMStabEB[row, column] = SystemsmatchEB[iteration]

################## PLOT OPTIONS ###############
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'beta']
EB = input('Exclude Boundaries? (y/n) ')
STAB = input('Stable Measures? (y/n) ')

##################### PLOT ###################
fig = plt.figure()

# Field Separation Plot
ax1 = fig.add_subplot(131)
ax1.set_xlabel('Strength of Initial Tectal Gradient')
ax1.set_ylabel('Receptive Field Separation')
ax1.set_xlim(0, 1)

# Field Size Plot
ax2 = fig.add_subplot(132)
ax2.set_xlabel('Strength of Initial Tectal Gradient')
ax2.set_ylabel('Receptive Field Size')
ax2.set_xlim(0, 1)

# Systems Match Plot
ax3 = fig.add_subplot(133)
ax3.set_xlabel('Strength of Initial Tectal Gradient')
ax3.set_ylabel('Systems Match')
ax3.set_xlim(0, 1)

if EB == 'y':
    if STAB == 'n':
        for betaval in range(2, 5):
            ax1.plot(yLT, SepEB[:, betaval], c=colours[betaval], label=beta[betaval])
        for betaval in range(2, 5):
            ax2.plot(yLT, SizeEB[:, betaval], c=colours[betaval], label=beta[betaval])
        for betaval in range(2, 5):
            ax3.plot(yLT, SMEB[:, betaval], c=colours[betaval], label=beta[betaval])

    elif STAB == 'y':
        for betaval in range(2, 5):
            ax1.plot(yLT, SepStabEB[:, betaval], c=colours[betaval], label=beta[betaval])
        for betaval in range(2, 5):
            ax2.plot(yLT, SizeStabEB[:, betaval], c=colours[betaval], label=beta[betaval])
        for betaval in range(2, 5):
            ax3.plot(yLT, SMStabEB[:, betaval], c=colours[betaval], label=beta[betaval])

elif EB == 'n':
    if STAB == 'n':
        for betaval in range(2, 5):
            ax1.plot(yLT, Sep[:, betaval], c=colours[betaval], label=beta[betaval])
        for betaval in range(2, 5):
            ax2.plot(yLT, Size[:, betaval], c=colours[betaval], label=beta[betaval])
        for betaval in range(2, 5):
            ax3.plot(yLT, SM[:, betaval], c=colours[betaval], label=beta[betaval])

    elif STAB == 'y':
        for betaval in range(2, 5):
            ax1.plot(yLT, SepStab[:, betaval], c=colours[betaval], label=beta[betaval])
        for betaval in range(2, 5):
            ax2.plot(yLT, SizeStab[:, betaval], c=colours[betaval], label=beta[betaval])
        for betaval in range(2, 5):
            ax3.plot(yLT, SMStab[:, betaval], c=colours[betaval], label=beta[betaval])

plt.legend()
plt.show()
