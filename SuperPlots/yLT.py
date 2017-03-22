import numpy as np
import matplotlib.pyplot as plt

#################### JOBS ###################

yLT = [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

JobID = [78, 77, 76, 74, 73, 72, 71, 70, 69, 68]

################# DATA MATRICES ###############

Sep = np.zeros([len(yLT)])
Size = np.zeros([len(yLT)])
SM = np.zeros([len(yLT)])

SepEB = np.zeros([len(yLT)])
SizeEB = np.zeros([len(yLT)])
SMEB = np.zeros([len(yLT)])

SepStab = np.zeros([len(yLT)])
SizeStab = np.zeros([len(yLT)])
SMStab = np.zeros([len(yLT)])

SepStabEB = np.zeros([len(yLT)])
SizeStabEB = np.zeros([len(yLT)])
SMStabEB = np.zeros([len(yLT)])

################### IMPORT DATA ################

for column in range(len(yLT)):
    ID = JobID[column]
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
        Sep[column] = Fieldseparation[-1]
        Size[column] = Fieldsize[-1]
        SM[column] = Systemsmatch[-1]

        SepEB[column] = FieldseparationEB[-1]
        SizeEB[column] = FieldsizeEB[-1]
        SMEB[column] = SystemsmatchEB[-1]

        SepStab[column] = Fieldseparation[iteration]
        SizeStab[column] = Fieldsize[iteration]
        SMStab[column] = Systemsmatch[iteration]

        SepStabEB[column] = FieldseparationEB[iteration]
        SizeStabEB[column] = FieldsizeEB[iteration]
        SMStabEB[column] = SystemsmatchEB[iteration]

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
        ax1.plot(yLT, SepEB)
        ax2.plot(yLT, SizeEB)
        ax3.plot(yLT, SMEB)

    elif STAB == 'y':
        ax1.plot(yLT, SepStabEB)
        ax2.plot(yLT, SizeStabEB)
        ax3.plot(yLT, SMStabEB)

elif EB == 'n':
    if STAB == 'n':
        ax1.plot(yLT, Sep)
        ax2.plot(yLT, Size)
        ax3.plot(yLT, SM)

    elif STAB == 'y':
        ax1.plot(yLT, SepStab)
        ax2.plot(yLT, SizeStab)
        ax3.plot(yLT, SMStab)

plt.show()
