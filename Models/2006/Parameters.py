JobID = 300  # JobID (or starting ID if multiple jobs)

# Iterations
Iterations = 100  # number of weight iterations
TRout = 10  # temporal resoultion of output files

# Layer sizes
NRdim1 = 10  # initial number of retinal cells
NRdim2 = 10
NTdim1 = 10  # initial number of tectal cells
NTdim2 = 10

# Tectal concentrations
alpha = 0.05
beta = 0.01
deltatc = 1  # deltaC time step
tc = 1  # concentration iterations per iteration

# Synaptic modification
gamma = 0.1
kappa = 0.0504
deltatw = 1  # deltaW time step
tw = 1  # weight iterations per iteration
distA = 1.  # the contribution made by EphA system to distance
distB = 1.

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
dRmindim2 = 1
dRmaxdim2 = 1  # NRdim2 // 4
dTmindim1 = 1
dTmaxdim1 = NTdim1 // 2
dTmindim2 = 1
dTmaxdim2 = 1  # NTdim2 // 4
dstep = 30  # time between growth iterations
td = 300  # time taken for new fibres to gain full strength
