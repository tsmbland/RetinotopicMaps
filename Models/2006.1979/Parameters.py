JobID = 0  # JobID (or starting ID if multiple jobs)

# Iterations
Iterations = 1  # number of weight iterations
TRout = 1  # temporal resoultion of output files

# Layer sizes
NRdim1 = 50  # initial number of retinal cells
NRdim2 = 50
NTdim1 = 50  # initial number of tectal cells
NTdim2 = 50

# Establishment of initial contacts
n0 = 10  # number of initial random contact
NLdim1 = NTdim1  # sets initial bias
NLdim2 = NTdim2

# Tectal concentrations
alpha = 0.05
beta = 0.05
deltatc = 1  # deltaC time step
tc = 1  # concentration iterations per iteration

# Synaptic modification
Wmax = 1.  # total strength available to each presynaptic fibre
gamma = 0.1
kappa = 0.5
k = 0.005
elim = 0.005  # elimination threshold
newW = 0.01  # weight of new synapses
sprout = 0.02  # sprouting threshold
deltatw = 1  # deltaW time step
tw = 1  # weight iterations per iteration
distA = 1.  # the contribution made by EphA system to distance
distB = 1.

# Mismatch surgery
sRmindim1 = NRdim1 // 2
sRmaxdim1 = NRdim1
sRmindim2 = 1
sRmaxdim2 = NRdim2
sTmindim1 = 1
sTmaxdim1 = NTdim1
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



# # Retinal Gradients
# y0Rdim1 = 1.0  # conc in cell 0
# ymRdim1 = 2.0  # conc in cell NRdim1/2
# ynRdim1 = 3.5  # conc in cell NRdim1
# y0Rdim2 = 0.1
# ymRdim2 = 0.5
# ynRdim2 = 1.0
# stochR = 0.
#
# # Tectal Gradients
# y0Tdim1 = 1. / y0Rdim1  # conc in cell 0 (before multiplication by yLT)
# ymTdim1 = 1. / ymRdim1  # conc in cell NTdim1/2
# ynTdim1 = 1. / ynRdim1  # conc in cell NTdim1
# y0Tdim2 = 0.1
# ymTdim2 = 0.5
# ynTdim2 = 1.0
# yLT = 1.0
# stochT = 0.
