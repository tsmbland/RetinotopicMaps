import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#################### JOBS ###################

yLT = [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

JobID = [78, 77, 76, 74, 73, 72, 71, 70, 69, 68]

################# OPTIONS ##################
sizechangethresh = 0.005
smthresh = 0.01

################# DATA MATRICES ###############

Sep = np.zeros([len(yLT)])
SepStdev = np.zeros([len(yLT)])
Size = np.zeros([len(yLT)])
SM = np.zeros([len(yLT)])

SepEB = np.zeros([len(yLT)])
SepStdevEB = np.zeros([len(yLT)])
SizeEB = np.zeros([len(yLT)])
SMEB = np.zeros([len(yLT)])

TStab = np.zeros([len(yLT)])
SepStab = np.zeros([len(yLT)])
SepStdevStab = np.zeros([len(yLT)])
SizeStab = np.zeros([len(yLT)])
SMStab = np.zeros([len(yLT)])

TStabEB = np.zeros([len(yLT)])
SepStabEB = np.zeros([len(yLT)])
SepStdevStabEB = np.zeros([len(yLT)])
SizeStabEB = np.zeros([len(yLT)])
SMStabEB = np.zeros([len(yLT)])

################### IMPORT DATA ################

for column in range(len(yLT)):
    ID = JobID[column]
    if ID != 0:
        # Import Data
        Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSize.npy' % ('{0:04}'.format(ID)))
        Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparation.npy' % ('{0:04}'.format(ID)))
        FieldseparationStdev = np.load('../../RetinotopicMapsData/%s/FieldSeparationStdev.npy' % ('{0:04}'.format(ID)))
        Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatch.npy' % ('{0:04}'.format(ID)))

        FieldsizeEB = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(ID)))
        FieldseparationEB = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(ID)))
        FieldseparationStdevEB = np.load('../../RetinotopicMapsData/%s/FieldSeparationStdevEB.npy' % ('{0:04}'.format(ID)))
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

        TRin = np.load('../../RetinotopicMapsData/%s/SecondaryTR.npy' % ('{0:04}'.format(ID)))

        Sep[column] = Fieldseparation[-1]
        SepStdev[column] = FieldseparationStdev[-1]
        Size[column] = Fieldsize[-1]
        SM[column] = Systemsmatch[-1]

        SepEB[column] = FieldseparationEB[-1]
        SepStdevEB[column] = FieldseparationStdevEB[-1]
        SizeEB[column] = FieldsizeEB[-1]
        SMEB[column] = SystemsmatchEB[-1]

        # Find stable time
        fieldsizechange = 1
        systemsmatchchange = 1
        iteration = 0
        while (fieldsizechange > sizechangethresh or fieldsizechange < -sizechangethresh or
                       systemsmatchchange > smthresh or systemsmatchchange < -smthresh) and iteration < len(
            Fieldsizechange):
            fieldsizechange = Fieldsizechange[iteration]
            systemsmatchchange = Systemsmatchchange[iteration]
            iteration += 1

        TStab[column] = iteration * TRin
        SepStab[column] = Fieldseparation[iteration]
        SepStdevStab[column] = FieldseparationStdev[iteration]
        SizeStab[column] = Fieldsize[iteration]
        SMStab[column] = Systemsmatch[iteration]

        # Find stable time (EB)
        fieldsizechange = 1
        systemsmatchchange = 1
        iteration = 0
        while (fieldsizechange > sizechangethresh or fieldsizechange < -sizechangethresh or
                       systemsmatchchange > smthresh or systemsmatchchange < -smthresh) and iteration < len(
            FieldsizechangeEB):
            fieldsizechange = FieldsizechangeEB[iteration]
            systemsmatchchange = SystemsmatchchangeEB[iteration]
            iteration += 1

        TStabEB[column] = iteration * TRin
        SepStabEB[column] = Fieldseparation[iteration]
        SepStdevStabEB[column] = FieldseparationStdev[iteration]
        SizeStabEB[column] = Fieldsize[iteration]
        SMStabEB[column] = Systemsmatch[iteration]

################## PLOT OPTIONS ###############
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
EB = input('Exclude Boundaries? (y/n) ')
STAB = input('Stable Measures? (y/n) ')

##################### PLOT ###################
fig = plt.figure()

# Field Separation Plot
ax1 = fig.add_subplot(321)
ax1.set_xlabel('Strength of Initial Tectal Gradient')
ax1.set_ylabel('Mean Receptive Field Separation')
ax1.set_xlim(0, 1)

# Field Separation Stdev Plot
ax2 = fig.add_subplot(322)
ax2.set_xlabel('Strength of Initial Tectal Gradient')
ax2.set_ylabel('Receptive Field Separation Standard Deviation')
ax2.set_xlim(0, 1)

# Field Size Plot
ax3 = fig.add_subplot(323)
ax3.set_xlabel('Strength of Initial Tectal Gradient')
ax3.set_ylabel('Mean Receptive Field Size')
ax3.set_xlim(0, 1)

# Systems Match Plot
ax4 = fig.add_subplot(324)
ax4.set_xlabel('Strength of Initial Tectal Gradient')
ax4.set_ylabel('Systems Match')
ax4.set_xlim(0, 1)

# Speed Plot
ax5 = fig.add_subplot(325)
ax5.set_xlabel('Strength of Initial Tectal Gradient')
ax5.set_ylabel('Stability Time')
ax5.set_xlim(0, 1)

if EB == 'y':
    if STAB == 'n':
        ax1.plot(yLT, SepEB)
        ax2.plot(yLT, SepStdevEB)
        ax3.plot(yLT, SizeEB)
        ax4.plot(yLT, SMEB)
        ax5.plot(yLT, TStabEB)

    elif STAB == 'y':
        ax1.plot(yLT, SepStabEB)
        ax2.plot(yLT, SepStdevStabEB)
        ax3.plot(yLT, SizeStabEB)
        ax4.plot(yLT, SMStabEB)
        ax5.plot(yLT, TStabEB)

elif EB == 'n':
    if STAB == 'n':
        ax1.plot(yLT, Sep)
        ax2.plot(yLT, SepStdev)
        ax3.plot(yLT, Size)
        ax4.plot(yLT, SM)
        ax5.plot(yLT, TStab)

    elif STAB == 'y':
        ax1.plot(yLT, SepStab)
        ax2.plot(yLT, SepStdevStab)
        ax3.plot(yLT, SizeStab)
        ax4.plot(yLT, SMStab)
        ax5.plot(yLT, TStab)

plt.show()
