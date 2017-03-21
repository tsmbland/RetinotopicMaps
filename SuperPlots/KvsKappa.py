import numpy as np
import matplotlib.pyplot as plt

#################### JOBS ###################

kappa = [0.0504, 0.1, 0.3, 0.5, 0.7, 1]
k = [0, 0.001, 0.005, 0.03]

JobID = np.zeros([len(kappa), len(k)], dtype=np.int8)
JobID[0, :] = [33, 59, 26, 46]
JobID[1, :] = [34, 60, 50, 47]
JobID[2, :] = [35, 61, 51, 48]
JobID[3, :] = [36, 62, 52, 49]
JobID[4, :] = [11, 17, 19, 9]
JobID[5, :] = [12, 14, 53, 13]

################# DATA MATRICES ###############

Sep = np.zeros([len(kappa), len(k)])
Size = np.zeros([len(kappa), len(k)])
SM = np.zeros([len(kappa), len(k)])

SepEB = np.zeros([len(kappa), len(k)])
SizeEB = np.zeros([len(kappa), len(k)])
SMEB = np.zeros([len(kappa), len(k)])

SepStab = np.zeros([len(kappa), len(k)])
SizeStab = np.zeros([len(kappa), len(k)])
SMStab = np.zeros([len(kappa), len(k)])

SepStabEB = np.zeros([len(kappa), len(k)])
SizeStabEB = np.zeros([len(kappa), len(k)])
SMStabEB = np.zeros([len(kappa), len(k)])

################### IMPORT DATA ################

for row in range(len(kappa)):
    for column in range(len(k)):
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
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
EB = input('Exclude Boundaries? (y/n) ')
STAB = input('Stable Measures? (y/n) ')

##################### PLOT ###################
fig = plt.figure()

# Field Separation Plot
ax1 = fig.add_subplot(131)
ax1.set_xlabel('Strength of Receptor-Ligand Interaction')
ax1.set_ylabel('Receptive Field Separation')
ax1.set_xlim(0, 1)

# Field Size Plot
ax2 = fig.add_subplot(132)
ax2.set_xlabel('Strength of Receptor-Ligand Interaction')
ax2.set_ylabel('Receptive Field Size')
ax2.set_xlim(0, 1)

# Systems Match Plot
ax3 = fig.add_subplot(133)
ax3.set_xlabel('Strength of Receptor-Ligand Interaction')
ax3.set_ylabel('Systems Match')
ax3.set_xlim(0, 1)

if EB == 'y':
    if STAB == 'n':
        for kval in range(len(k)):
            ax1.plot(kappa, SepEB[:, kval], c=colours[kval], label=k[kval])
        for kval in range(len(k)):
            ax2.plot(kappa, SizeEB[:, kval], c=colours[kval], label=k[kval])
        for kval in range(len(k)):
            ax3.plot(kappa, SMEB[:, kval], c=colours[kval], label=k[kval])

    elif STAB == 'y':
        for kval in range(len(k)):
            ax1.plot(kappa, SepStabEB[:, kval], c=colours[kval], label=k[kval])
        for kval in range(len(k)):
            ax2.plot(kappa, SizeStabEB[:, kval], c=colours[kval], label=k[kval])
        for kval in range(len(k)):
            ax3.plot(kappa, SMStabEB[:, kval], c=colours[kval], label=k[kval])

elif EB == 'n':
    if STAB == 'n':
        for kval in range(len(k)):
            ax1.plot(kappa, Sep[:, kval], c=colours[kval], label=k[kval])
        for kval in range(len(k)):
            ax2.plot(kappa, Size[:, kval], c=colours[kval], label=k[kval])
        for kval in range(len(k)):
            ax3.plot(kappa, SM[:, kval], c=colours[kval], label=k[kval])

    elif STAB == 'y':
        for kval in range(len(k)):
            ax1.plot(kappa, SepStab[:, kval], c=colours[kval], label=k[kval])
        for kval in range(len(k)):
            ax2.plot(kappa, SizeStab[:, kval], c=colours[kval], label=k[kval])
        for kval in range(len(k)):
            ax3.plot(kappa, SMStab[:, kval], c=colours[kval], label=k[kval])

plt.legend()
plt.show()
