import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#################### JOBS ###################

stochR = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
beta = [0, 0.001, 0.01, 0.1, 0.2, 0.5, 1.]

JobID = np.zeros([len(stochR), len(beta)], dtype=np.int32)
JobID[0, :] = [0, 0, 79, 80, 81, 82, 83]
JobID[1, :] = [0, 0, 136, 141, 146, 151, 0]
JobID[2, :] = [0, 0, 137, 142, 147, 152, 0]
JobID[3, :] = [0, 0, 138, 143, 148, 153, 0]
JobID[4, :] = [0, 0, 139, 144, 149, 154, 0]
JobID[5, :] = [0, 0, 140, 145, 150, 155, 0]

################# OPTIONS ##################
sizechangethresh = 0.005
smthresh = 0.01

################# DATA MATRICES ###############

Sep = np.zeros([len(stochR), len(beta)])
SepStdev = np.zeros([len(stochR), len(beta)])
Size = np.zeros([len(stochR), len(beta)])
SM = np.zeros([len(stochR), len(beta)])

SepEB = np.zeros([len(stochR), len(beta)])
SepStdevEB = np.zeros([len(stochR), len(beta)])
SizeEB = np.zeros([len(stochR), len(beta)])
SMEB = np.zeros([len(stochR), len(beta)])

TStab = np.zeros([len(stochR), len(beta)])
SepStab = np.zeros([len(stochR), len(beta)])
SepStdevStab = np.zeros([len(stochR), len(beta)])
SizeStab = np.zeros([len(stochR), len(beta)])
SMStab = np.zeros([len(stochR), len(beta)])

TStabEB = np.zeros([len(stochR), len(beta)])
SepStabEB = np.zeros([len(stochR), len(beta)])
SepStdevStabEB = np.zeros([len(stochR), len(beta)])
SizeStabEB = np.zeros([len(stochR), len(beta)])
SMStabEB = np.zeros([len(stochR), len(beta)])

################### IMPORT DATA ################

for row in range(len(stochR)):
    for column in range(1, 6):
        ID = JobID[row, column]
        if ID != 0:
            # Import Data
            Fieldsize = np.load('../../RetinotopicMapsData/%s/FieldSize.npy' % ('{0:04}'.format(ID)))
            Fieldseparation = np.load('../../RetinotopicMapsData/%s/FieldSeparation.npy' % ('{0:04}'.format(ID)))
            FieldseparationStdev = np.load(
                '../../RetinotopicMapsData/%s/FieldSeparationStdev.npy' % ('{0:04}'.format(ID)))
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

            Sep[row, column] = Fieldseparation[-1]
            SepStdev[row, column] = FieldseparationStdev[-1]
            Size[row, column] = Fieldsize[-1]
            SM[row, column] = Systemsmatch[-1]

            SepEB[row, column] = FieldseparationEB[-1]
            SepStdevEB[row, column] = FieldseparationStdevEB[-1]
            SizeEB[row, column] = FieldsizeEB[-1]
            SMEB[row, column] = SystemsmatchEB[-1]

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

            TStab[row, column] = iteration * TRin
            SepStab[row, column] = Fieldseparation[iteration]
            SepStdevStab[row, column] = FieldseparationStdev[iteration]
            SizeStab[row, column] = Fieldsize[iteration]
            SMStab[row, column] = Systemsmatch[iteration]

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

            TStabEB[row, column] = iteration * TRin
            SepStabEB[row, column] = Fieldseparation[iteration]
            SepStdevStabEB[row, column] = FieldseparationStdev[iteration]
            SizeStabEB[row, column] = Fieldsize[iteration]
            SMStabEB[row, column] = Systemsmatch[iteration]

################## PLOT OPTIONS ###############
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
EB = input('Exclude Boundaries? (y/n) ')
STAB = input('Stable Measures? (y/n) ')

##################### PLOT ###################
fig = plt.figure()

# Field Separation Plot
ax1 = fig.add_subplot(321)
ax1.set_xlabel('Retinal Gradient Stochasticity')
ax1.set_ylabel('Mean Receptive Field Separation')
ax1.set_xlim(0, 0.5)

# Field Separation Stdev Plot
ax2 = fig.add_subplot(322)
ax2.set_xlabel('Retinal Gradient Stochasticity')
ax2.set_ylabel('Receptive Field Separation Standard Deviation')
ax2.set_xlim(0, 0.5)

# Field Size Plot
ax3 = fig.add_subplot(323)
ax3.set_xlabel('Retinal Gradient Stochasticity')
ax3.set_ylabel('Mean Receptive Field Size')
ax3.set_xlim(0, 0.5)

# Systems Match Plot
ax4 = fig.add_subplot(324)
ax4.set_xlabel('Retinal Gradient Stochasticity')
ax4.set_ylabel('Systems Match')
ax4.set_xlim(0, 0.5)

# Systems Match Plot
ax4 = fig.add_subplot(324)
ax4.set_xlabel('Retinal Gradient Stochasticity')
ax4.set_ylabel('Systems Match')
ax4.set_xlim(0, 0.5)

# Systems Match Plot
ax5 = fig.add_subplot(325)
ax5.set_xlabel('Retinal Gradient Stochasticity')
ax5.set_ylabel('Stability Time')
ax5.set_xlim(0, 0.5)

if EB == 'y':
    if STAB == 'n':
        for betaval in range(1, 6):
            ax1.plot(stochR, SepEB[:, betaval], c=colours[betaval], label=beta[betaval])
            ax2.plot(stochR, SepStdevEB[:, betaval], c=colours[betaval], label=beta[betaval])
            ax3.plot(stochR, SizeEB[:, betaval], c=colours[betaval], label=beta[betaval])
            ax4.plot(stochR, SMEB[:, betaval], c=colours[betaval], label=beta[betaval])
            ax5.plot(stochR, TStabEB[:, betaval], c=colours[betaval], label=beta[betaval])

    elif STAB == 'y':
        for betaval in range(1, 6):
            ax1.plot(stochR, SepStabEB[:, betaval], c=colours[betaval], label=beta[betaval])
            ax2.plot(stochR, SepStdevStabEB[:, betaval], c=colours[betaval], label=beta[betaval])
            ax3.plot(stochR, SizeStabEB[:, betaval], c=colours[betaval], label=beta[betaval])
            ax4.plot(stochR, SMStabEB[:, betaval], c=colours[betaval], label=beta[betaval])
            ax5.plot(stochR, TStabEB[:, betaval], c=colours[betaval], label=beta[betaval])

elif EB == 'n':
    if STAB == 'n':
        for betaval in range(1, 6):
            ax1.plot(stochR, Sep[:, betaval], c=colours[betaval], label=beta[betaval])
            ax2.plot(stochR, SepStdev[:, betaval], c=colours[betaval], label=beta[betaval])
            ax3.plot(stochR, Size[:, betaval], c=colours[betaval], label=beta[betaval])
            ax4.plot(stochR, SM[:, betaval], c=colours[betaval], label=beta[betaval])
            ax5.plot(stochR, TStab[:, betaval], c=colours[betaval], label=beta[betaval])

    elif STAB == 'y':
        for betaval in range(1, 6):
            ax1.plot(stochR, SepStab[:, betaval], c=colours[betaval], label=beta[betaval])
            ax2.plot(stochR, SepStdevStab[:, betaval], c=colours[betaval], label=beta[betaval])
            ax3.plot(stochR, SizeStab[:, betaval], c=colours[betaval], label=beta[betaval])
            ax4.plot(stochR, SMStab[:, betaval], c=colours[betaval], label=beta[betaval])
            ax5.plot(stochR, TStab[:, betaval], c=colours[betaval], label=beta[betaval])

plt.legend()
plt.show()
