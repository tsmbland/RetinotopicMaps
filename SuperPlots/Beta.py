import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#################### JOBS ###################

Beta = [0.001, 0.01, 0.02, 0.05, 0.07, 0.1, 1]

JobID = [306, 307, 308, 309, 310, 311, 312]

################# OPTIONS ##################
sizechangethresh = 0.005
smthresh = 0.01

################# DATA MATRICES ###############

Sep = np.zeros([len(Beta)])
SepStdev = np.zeros([len(Beta)])
Size = np.zeros([len(Beta)])
SM = np.zeros([len(Beta)])

SepEB = np.zeros([len(Beta)])
SepStdevEB = np.zeros([len(Beta)])
SizeEB = np.zeros([len(Beta)])
SMEB = np.zeros([len(Beta)])

TStab = np.zeros([len(Beta)])
SepStab = np.zeros([len(Beta)])
SepStdevStab = np.zeros([len(Beta)])
SizeStab = np.zeros([len(Beta)])
SMStab = np.zeros([len(Beta)])

TStabEB = np.zeros([len(Beta)])
SepStabEB = np.zeros([len(Beta)])
SepStdevStabEB = np.zeros([len(Beta)])
SizeStabEB = np.zeros([len(Beta)])
SMStabEB = np.zeros([len(Beta)])

################### IMPORT DATA ################

for column in range(len(Beta)):
    ID = JobID[column]
    if ID != 0:
        # Import Data
        Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSize.npy' % ('{0:04}'.format(ID)))
        Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparation.npy' % ('{0:04}'.format(ID)))
        FieldseparationStdev = np.load('../../RetinotopicMapsData/%s/FieldSeparationStdev.npy' % ('{0:04}'.format(ID)))
        Systemsmatch = np.load('../../RetinotopicMapsData/%s/SystemsMatch.npy' % ('{0:04}'.format(ID)))

        FieldsizeEB = np.load('../../RetinotopicMapsData/%s/FieldSizeEB.npy' % ('{0:04}'.format(ID)))
        FieldseparationEB = np.load('../../RetinotopicMapsData/%s/FieldSeparationEB.npy' % ('{0:04}'.format(ID)))
        FieldseparationStdevEB = np.load(
            '../../RetinotopicMapsData/%s/FieldSeparationStdevEB.npy' % ('{0:04}'.format(ID)))
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
EB = input('Exclude Boundaries? (y/n) ')
STAB = input('Stable Measures? (y/n) ')

##################### PLOT ###################
fig = plt.figure()

# Field Separation Plot
ax1 = fig.add_subplot(321)
ax1.set_xlabel('Beta')
ax1.set_ylabel('Mean Receptive Field Separation')
ax1.set_xlim(0, 1)

# Field Separation Stdev Plot
ax2 = fig.add_subplot(322)
ax2.set_xlabel('Beta')
ax2.set_ylabel('Receptive Field Separation Standard Deviation')
ax2.set_xlim(0, 1)

# Field Size Plot
ax3 = fig.add_subplot(323)
ax3.set_xlabel('Beta')
ax3.set_ylabel('Mean Receptive Field Size')
ax3.set_xlim(0, 1)

# Systems Match Plot
ax4 = fig.add_subplot(324)
ax4.set_xlabel('Beta')
ax4.set_ylabel('Systems Match')
ax4.set_xlim(0, 1)

# Speed Plot
ax5 = fig.add_subplot(325)
ax5.set_xlabel('Beta')
ax5.set_ylabel('Stability Time')
ax5.set_xlim(0, 1)

if EB == 'y':
    if STAB == 'n':
        ax1.plot(Beta, SepEB)
        ax2.plot(Beta, SepStdevEB)
        ax3.plot(Beta, SizeEB)
        ax4.plot(Beta, SMEB)
        ax5.plot(Beta, TStabEB)

    elif STAB == 'y':
        ax1.plot(Beta, SepStabEB)
        ax2.plot(Beta, SepStdevStabEB)
        ax3.plot(Beta, SizeStabEB)
        ax4.plot(Beta, SMStabEB)
        ax5.plot(Beta, TStabEB)

elif EB == 'n':
    if STAB == 'n':
        ax1.plot(Beta, Sep)
        ax2.plot(Beta, SepStdev)
        ax3.plot(Beta, Size)
        ax4.plot(Beta, SM)
        ax5.plot(Beta, TStab)

    elif STAB == 'y':
        ax1.plot(Beta, SepStab)
        ax2.plot(Beta, SepStdevStab)
        ax3.plot(Beta, SizeStab)
        ax4.plot(Beta, SMStab)
        ax5.plot(Beta, TStab)

plt.show()
