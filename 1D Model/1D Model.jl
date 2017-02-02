starttime = time()
#################### PARAMETERS #####################

# General
const NR = 80  # initial number of retinal cells
const NT = 80  # initial number of tectal cells
const M = 7  # number of markers

# Presynaptic concentrations
const a = 0.006  # (or 0.003) #decay constant
const d = 0.3  # diffusion length constant
const E = 0.01  # synaptic elimination threshold
const Q = 100  # release of markers from source
const stab = 0.1  # retinal stability threshold

# Establishment of initial contacts
const n0 = 8  # number of initial random contact
const NL = 60  # sets initial bias

# Tectal concentrations
const deltat = 1  # time step
const td = 5  # number of concentration iterations per weight iteration

# Synaptic modification
const W = 1  # total strength available to each presynaptic fibre
const h = 0.01  # ???
const k = 0.03  # ???
const elim = 0.005  # elimination threshold
const Iterations = 10  # number of weight iterations

const pad = 1  #  zeros pad size

###############  VARIABLES ###################
const rmin = 1
const rmax = NR
const tmin = 1
const tmax = NT
const nR = rmax - rmin + 1  # present number of retinal cells (pre-surgery)
const nT = tmax - tmin + 1  # present number of tectal cells (pre-surgery)

Wpt = zeros(Float64, (NT + 2*pad, NR + 2*pad))  # synaptic strength between a presynaptic cell and a postsynaptic cell
Qpm = zeros(Float64, (NR + 2*pad, M))  # presence of marker sources along retina
Qtm = zeros(Float64, (NT + 2*pad, M))  # axonal flow of molecules into postsymaptic cells
Cpm = zeros(Float64, (NR + 2*pad, M))  # concentration of a molecule in a presynaptic cell
Ctm = zeros(Float64, (NT + 2*pad, M))  # concentration of a molecule in a postsynaptic cell
normalisedCpm = zeros(Float64, (NR + 2*pad, M))  # normalised (by marker conc.) marker concentration  in a presynaptic cell
normalisedCtm = zeros(Float64, (NT + 2*pad, M))  # normalised (by marker conc.) marker concentration in a postsynaptic cell

################## RETINA #####################


# MARKER LOCATIONS
markerspacing = NR / (M - 1)
location = 1.0
for m in 1:M-1
  Qpm[round(Int, location) + pad, m] = Q
  location += markerspacing
  Qpm[NR + pad, M] = Q
end

# PRESYNAPTIC CONCENTRATIONS
function conc_change(concmatrix, layer)
  # Layer
  if layer == "presynaptic"
    Qmatrix = Qpm
    layermin = rmin
    layermax = rmax
  elseif layer == "tectal"
    Qmatrix = Qtm
    layermin = tmin
    layermax = tmax
  end

  # Neuron map
  nm = zeros(Int,length(concmatrix[:, 1]))
  nm[layermin + pad: layermax + pad] = 1

  # Neighbour Count
  nc = zeros(Int,length(concmatrix[:, 1]))
  for cell in layermin+pad:layermax+pad
    nc[cell] = nm[cell - 1] + nm[cell + 1]
  end

  # Conc change
  concchange = zeros(length(concmatrix[:, 1]), M)
  for m in 1:M
    for cell in layermin+pad:layermax+pad
      concchange[cell, m] = (-a * concmatrix[cell, m] + d * (
      concmatrix[cell - 1, m] - nc[cell] * concmatrix[cell, m] + concmatrix[cell + 1, m]) + Qmatrix[cell, m])
    end
  end

  return concchange
end

averagemarkerchange = 1
while averagemarkerchange > stab
  deltaconc = conc_change(Cpm, "presynaptic")
  averagemarkerchange = (sum(sum(deltaconc)) / sum(sum(Cpm))) * 100
  Cpm += (deltaconc * deltat)
end

# NORMALISED PRESYNAPTIC CONCENTRATIONS
function normalise(concmatrix, layer)
  # Layer
  if layer == "presynaptic"
    layermin = rmin
    layermax = rmax
  elseif layer == "tectal"
    layermin = tmin
    layermax = tmax
  end

  # Normalisation
  normalised = zeros(length(concmatrix[:, 1]), M)
  markersum = zeros(length(concmatrix[:, 1]))
  for cell in layermin:layermax
    markersum[cell+pad] = sum(concmatrix[cell+pad, :])
  end

  for m in 1:M
    for cell in layermin:layermax
      normalised[cell+pad, m] = concmatrix[cell+pad, m] / markersum[cell+pad]
      if normalised[cell+pad, m] < E
        normalised[cell+pad, m] = 0
      end
    end
  end

  return normalised
end


normalisedCpm = normalise(Cpm, "presynaptic")


#################### CONNECTIONS ######################


# INITIAL CONNECTIONS
function initialconnections(p::Int64)
  # p = presynaptic cell number
  initialstrength = W / n0
  arrangement = zeros(NL)
  arrangement[1:n0] = initialstrength

  if floor(p * ((NT - NL) / NR) + NL) <= tmax
    shuffle(arrangement)
    Wpt[floor(Int, p * ((NT - NL) / NR)) + 1 + pad: floor(Int, p * ((NT - NL) / NR) + NL) + pad, p + pad] = arrangement
  else
    shrunkarrangement = zeros(tmax - floor(p * ((NT - NL) / NR)))
    shrunkarrangement[1:n0] = initialstrength
    shuffle(shrunkarrangement)
    Wpt[floor(Int, p * ((NT - NL) / NR)) + 1 + pad: tmax + pad, p + pad] = shrunkarrangement
  end
end

for p in rmin:rmax
  initialconnections(p)
end

# INITIAL CONCENTRATIONS
function updateQtm()
  Qtm[:,:] = 0.
  for tectal in tmin+pad:tmax+pad
    for p in rmin+pad:rmax+pad
      for m in 1:M
        Qtm[tectal, m] += normalisedCpm[p,m] * Wpt[tectal, p]
      end
    end
  end
end

updateQtm()
for t in 1:td
  deltaconc = conc_change(Ctm, "tectal")
  Ctm += (deltaconc * deltat)
end
normalisedCtm = normalise(Ctm, "tectal")


# ITERATIONS

function weight_change()
  # SYNAPTIC WEIGHT

  newweight = zeros(NT + 2*pad, NR + 2*pad)
  for p in rmin+pad:rmax+pad
    totalSp = 0
    connections = 0
    deltaWsum = 0
    deltaWpt = zeros(NT + 2*pad)
    Spt = zeros(NT + 2*pad)

    for tectal in tmin+pad:tmax+pad

      # Calculate similarity
      for m in 1:M
        Spt[tectal] += minimum((normalisedCpm[p, m], normalisedCtm[tectal, m]))
      end

      # Count connections
      if Wpt[tectal, p] > 0
        totalSp += Spt[tectal]
        connections += 1
      end
    end

    # Calculate mean similarity
    meanSp = (totalSp / connections) - k

    for tectal in tmin+pad:tmax+pad

      # Calculate deltaW
      deltaWpt[tectal] = h * (Spt[tectal] - meanSp)

      # Calculate deltaWsum
      if Wpt[tectal, p] > 0
        deltaWsum += deltaWpt[tectal]
      end
    end

    for tectal in tmin+pad:tmax+pad

      # Calculate new W
      newweight[tectal, p] = (Wpt[tectal, p] + deltaWpt[tectal]) * W / (W + deltaWsum)

      # REMOVE SYNAPSES
      if newweight[tectal, p] < elim * W
        newweight[tectal, p] = 0
      end
    end

    # ADD NEW SYNAPSES
    for tectal in tmin+pad:tmax+pad
      if newweight[tectal, p] == 0 && (newweight[tectal + 1, p] > 0.02 * W || newweight[tectal - 1, p] > 0.02 * W)
        newweight[tectal, p] = 0.01 * W
      end
    end
  end

  # CALCULATE WEIGHT CHANGE
  weightchange = newweight - Wpt
  return weightchange
end

for iterations in 1:Iterations
  deltaW = weight_change()
  Wpt += deltaW

  updateQtm()
  for t in 1:td
    deltaconc = conc_change(Ctm, "tectal")
    Ctm += (deltaconc * deltat)
  end
  normalisedCtm = normalise(Ctm, "tectal")
end


timeelapsed = time() - starttime
print("Time elapsed: ", timeelapsed, " seconds")
