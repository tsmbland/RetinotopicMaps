JobID = 0  # JobID (or starting ID if multiple jobs)

# Iterations
Iterations = 500  # number of weight iterations
TRout = 20  # temporal resoultion of output files

# Layer sizes
NRdim1 = 20  # initial number of retinal cells (nasal-temporal)
NRdim2 = 20  # initial number of retinal cells (dorsal-ventral)
NTdim1 = 20  # initial number of tectal cells (posterior-anterior)
NTdim2 = 20  # initial number of tectal cells (lateral-medial)

# Tectal concentrations
alpha = 0.05  # strength of marker induction
beta = 0.01  # strength of neighbour communication
deltatc = 1  # concentration update time step
tc = 1  # concentration iterations per iteration

# Synaptic modification
gamma = 0.1  # rate of weight update
kappa = 0.0504  # sharpness of receptor-ligand comparison
deltatw = 1  # weight update time step
tw = 1  # weight iterations per iteration
distA = 1.  # the contribution made by EphA system to distance (set to 0 in certain 1D simulations)
distB = 1.  # the contribution made by EphB system to distance (set to 0 in certain 1D simulations)

# Mismatch surgery
sRmindim1 = 1
sRmaxdim1 = NRdim1
sRmindim2 = 1
sRmaxdim2 = NRdim2
sTmindim1 = 1
sTmaxdim1 = NTdim1 // 2
sTmindim2 = 1
sTmaxdim2 = NTdim2

# Development
dRmindim1 = NRdim1 // 4
dRmaxdim1 = 3 * NRdim1 // 4
dRmindim2 = NRdim2 // 4
dRmaxdim2 = 3 * NRdim2 // 4
dTmindim1 = 1
dTmaxdim1 = NTdim1 // 2
dTmindim2 = 1
dTmaxdim2 = NTdim2 // 2
dstep = 30  # time between growth updates
td = 300  # time taken for new fibres to gain full strength
