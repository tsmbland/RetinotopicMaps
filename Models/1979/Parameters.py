# Iterations
Iterations = 500  # number of weight iterations
TRout = 5  # temporal resoultion of output files

# Layer Sizes
NRdim1 = 20  # initial number of retinal cells
NRdim2 = 20
NTdim1 = 20  # initial number of tectal cells
NTdim2 = 20

# Markers
Mdim1 = 3  # number of markers
Mdim2 = 3

# Establishment of initial contacts
n0 = 7  # number of initial random contact
NLdim1 = 15  # sets initial bias
NLdim2 = 15

# Presynaptic concentrations
a = 0.006  # (or 0.003) #decay constant
d = 0.3  # diffusion length constant
E = 0.01  # concentration elimination threshold
Q = 100.  # release of markers from source
stab = 0.1  # retinal stability threshold

# Tectal concentrations
deltat = 0.5  # time step
tc = 10  # number of concentration iterations per weight iteration

# Synaptic modification
Wmax = 1.  # total (final) strength available to each presynaptic fibre
h = 0.01  # ???
k = 0.03  # ???
elim = 0.005  # elimination threshold
newW = 0.01  # weight of new synapses
sprout = 0.02  # sprouting threshold

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
dstep = 30  # time between growth iterations
td = 300  # time taken for new fibres to gain full strength