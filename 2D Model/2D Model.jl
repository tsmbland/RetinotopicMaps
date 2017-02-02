starttime = time()
###################### MODEL ########################


#################### PARAMETERS #####################

# General
Iterations = 50  # number of weight iterations
NRdim1 = 80  # initial number of retinal cells
NRdim2 = 1
NTdim1 = 80  # initial number of tectal cells
NTdim2 = 1
Mdim1 = 7  # number of markers
Mdim2 = 1

# Establishment of initial contacts
n0 = 7  # number of initial random contact
NLdim1 = 60   # sets initial bias
NLdim2 = 1

# Presynaptic concentrations
a = 0.006  # (or 0.003) #decay constant
d = 0.3  # diffusion length constant
E = 0.01  # concentration elimination threshold
Q = 100.  # release of markers from source
stab = 0.1  # retinal stability threshold

# Tectal concentrations
deltat = 0.5  # time step
td = 10  # number of concentration iterations per weight iteration

# Synaptic modification
W = 1.  # total strength available to each presynaptic fibre
h = 0.01  # ???
k = 0.03  # ???
elim = 0.005  # elimination threshold
newW = 0.01  # weight of new synapses
sprout = 0.02  # sprouting threshold

pad = 1

################### VARIABLES ###################
rmindim1 = 1
rmaxdim1 = NRdim1
rmindim2 = 1
rmaxdim2 = NRdim2
tmindim1 = 1
tmaxdim1 = NTdim1
tmindim2 = 1
tmaxdim2 = NTdim2

nRdim1 = rmaxdim1 - rmindim1 + 1  # present number of retinal cells (pre-surgery)
nRdim2 = rmaxdim2 - rmindim2 + 1
nTdim1 = tmaxdim1 - tmindim1 + 1  # present number of tectal cells (pre-surgery)
nTdim2 = tmaxdim2 - tmindim2 + 1
M = Mdim1 * Mdim2

Wpt = zeros(Float64, (Iterations + 1, NTdim1 + 2*pad, NTdim2 + 2*pad, NRdim1 + 2*pad,
NRdim2 + 2*pad))  # synaptic strength between a presynaptic cell and a postsynaptic cell

Qpm = zeros(Float64, (M, NRdim1 + 2*pad, NRdim2 + 2*pad))  # presence of marker sources along retina
Qtm = zeros(Float64, (M, NTdim1 + 2*pad, NTdim2 + 2*pad))  # axonal flow of molecules into postsymaptic cells

Cpm = zeros(Float64, (M, NRdim1 + 2*pad, NRdim2 + 2*pad))  # concentration of a molecule in a presynaptic cell
Ctm = zeros(Float64, (M, NTdim1 + 2*pad, NTdim2 + 2*pad))  # concentration of a molecule in a postsynaptic cell
normalisedCpm = zeros(Float64,
(M, NRdim1 + 2*pad, NRdim2 + 2*pad))  # normalised (by marker conc.) marker concentration  in a presynaptic cell
normalisedCtm = zeros(Float64,
(M, NTdim1 + 2*pad, NTdim2 + 2*pad))  # normalised (by marker conc.) marker concentration in a postsynaptic cell

NCp = zeros(Int, (NRdim1 + 2*pad, NRdim2 + 2*pad))
NCt = zeros(Int, (NTdim1 + 2*pad, NTdim2 + 2*pad))

currentiteration = 1

################### FUNCTIONS #####################

function updateNc()
  # Presynaptic neuron map
  nmp = zeros(Int, (NRdim1 + 2*pad, NRdim2 + 2*pad))
  nmp[rmindim1 + pad:rmaxdim1 + pad, rmindim2+pad:rmaxdim2 + pad] = 1

  # Tectal neuron map
  nmt = zeros(Int, (NTdim1 + 2*pad, NTdim2 + 2*pad))
  nmt[tmindim1+pad:tmaxdim1 + pad, tmindim2+pad:tmaxdim2 + pad] = 1

  # Retinal neighbour count
  for rdim1 in rmindim1 + pad : rmaxdim1 + pad
    for rdim2 in rmindim2 + pad : rmaxdim2 + pad
      NCp[rdim1, rdim2] = nmp[rdim1 + 1, rdim2] + nmp[rdim1 - 1, rdim2] + nmp[rdim1, rdim2 + 1] + nmp[
      rdim1, rdim2 - 1]
    end
  end

  # Tectal neighbour count
  for tdim1 in tmindim1 + pad : tmaxdim1 + pad
    for tdim2 in tmindim2 + pad : tmaxdim2 + pad
      NCt[tdim1, tdim2] = nmt[tdim1 + 1, tdim2] + nmt[tdim1 - 1, tdim2] + nmt[tdim1, tdim2 + 1] + nmt[tdim1, tdim2 - 1]
    end
  end
end



function conc_change(concmatrix, layer)
  # Layer
  if layer == "presynaptic"
    dim1start = rmindim1
    dim1end = rmaxdim1
    dim2start = rmindim2
    dim2end = rmaxdim2
    Qmatrix = Qpm
    nc = NCp
  elseif layer == "tectal"
    dim1start = tmindim1
    dim1end = tmaxdim1
    dim2start = tmindim2
    dim2end = tmaxdim2
    Qmatrix = Qtm
    nc = NCt
  end

  # Conc change
  concchange = zeros(Float64, (M, length(concmatrix[1, :, 1]), length(concmatrix[1, 1, :])))
  for m in 1:M
    for dim1 in dim1start + pad : dim1end + pad
      for dim2 in dim2start + pad : dim2end + pad
        concchange[m, dim1, dim2] = (-a * concmatrix[m, dim1, dim2] + d * (
        concmatrix[m, dim1, dim2 + 1] + concmatrix[m, dim1, dim2 - 1] + concmatrix[m, dim1 + 1, dim2] +
        concmatrix[m, dim1 - 1, dim2] - nc[dim1, dim2] * concmatrix[m, dim1, dim2]) + Qmatrix[
        m, dim1, dim2])
      end
    end
  end
  return concchange
end

function normalise(concmatrix, layer)
  # Layer
  if layer == "presynaptic"
    dim1start = rmindim1
    dim1end = rmaxdim1
    dim2start = rmindim1
    dim2end = rmaxdim2
  elseif layer == "tectal"
    dim1start = tmindim1
    dim1end = tmaxdim1
    dim2start = tmindim1
    dim2end = tmaxdim2
  end

  # Matrix size
  lengthdim1 = length(concmatrix[1, :, 1])
  lengthdim2 = length(concmatrix[1, 1, :])

  # Marker sum
  markersum = zeros(Float64, (lengthdim1, lengthdim2))
  for dim1 in dim1start + pad : dim1end + pad
    for dim2 in dim2start + pad : dim2end + pad
      markersum[dim1, dim2] = sum(concmatrix[:, dim1, dim2])
    end
  end


  # Normalisation
  normalised = zeros(Float64, (M, lengthdim1, lengthdim2))

  for m in 1:M
    for dim1 in dim1start + pad : dim1end + pad
      for dim2 in dim2start + pad : dim2end + pad
        normalised[m, dim1, dim2] = concmatrix[m, dim1, dim2] / markersum[dim1, dim2]
        if normalised[m, dim1, dim2] < E
          normalised[m, dim1, dim2] = 0
        end
      end
    end
  end

  return normalised
end

function initialconections(rdim1, rdim2)
  initialstrength = W / n0
  if floor(rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1) <= nTdim1
    if floor(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= nTdim2
      # Fits in both dimensions
      arrangement = zeros(Float64, (NLdim1 * NLdim2))
      arrangement[1:n0] = initialstrength
      shuffle(arrangement)
      arrangement = reshape(arrangement, (NLdim1, NLdim2))
      Wpt[currentiteration, floor(Int, rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: floor(Int,
      rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1),
      floor(Int, rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: floor(Int,
      rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2), rdim1,
      rdim2] = arrangement
    else
      # Fits in dim1 but not dim2
      arrangement = zeros(Float64,((NTdim2 - floor(Int, rdim2 * ((NTdim2 - NLdim2) / NRdim2))) * NLdim1))
      arrangement[1:n0] = initialstrength
      shuffle(arrangement)
      arrangement = reshape(arrangement, (NLdim1, NTdim2 - floor(Int, rdim2 * ((NTdim2 - NLdim2) / NRdim2))))
      Wpt[currentiteration, floor(Int, rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: floor(Int,
      rdim1 * ((NTdim1 - NLdim1) / NRdim1) + NLdim1),
      floor(Int, rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: NTdim2, rdim1,
      rdim2] = arrangement
    end

  elseif int(rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2) <= nTdim2
    # Doesn't fit into dim1 but fits into dim2
    arrangement = zeros(Float64, (NTdim1 - floor(Int, rdim1 * ((NTdim1 - NLdim1) / NRdim1))) * NLdim2)
    arrangement[1:n0] = initialstrength
    shuffle(arrangement)
    arrangement = reshape(arrangement, (NTdim1 - floor(Int, rdim1 * ((NTdim1 - NLdim1) / NRdim1)), NLdim2))
    Wpt[currentiteration, floor(Int, rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: NTdim1,
    floor(Int, rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: floor(Int, rdim2 * ((NTdim2 - NLdim2) / NRdim2) + NLdim2),
    rdim1,
    rdim2] = arrangement
  else
    # Doesn't fit into either dimension
    arrangement = zeros(Float64, (NTdim1 - floor(Int, rdim1 * ((NTdim1 - NLdim1) / NRdim1))) * (
    NTdim2 - floor(Int, rdim2 * ((NTdim2 - NLdim2) / NRdim2))))
    arrangement[1:n0] = initialstrength
    shuffle(arrangement)
    arrangement = reshape(arrangement, (
    NTdim1 - floor(Int, rdim1 * ((NTdim1 - NLdim1) / NRdim1)),
    NTdim2 - floor(Int, rdim2 * ((NTdim2 - NLdim2) / NRdim2))))
    Wpt[currentiteration, floor(Int, rdim1 * ((NTdim1 - NLdim1) / NRdim1)) + 1: NTdim1,
    floor(Int, rdim2 * ((NTdim2 - NLdim2) / NRdim2)) + 1: NTdim2,
    rdim1,
    rdim2] = arrangement
  end
end

function updateQtm()
  Qtm[:, :, :] = 0.
  for tdim1 in tmindim1+pad : tmaxdim1 + pad
    for tdim2 in tmindim2+pad : tmaxdim2 + pad
      for m in 1:M
        Qtm[m, tdim1, tdim2] = sum(sum(normalisedCpm[m, :, :] .* Wpt[
        currentiteration, tdim1, tdim2, :, :]))
      end
    end
  end
end

function updateWeight()
  # SYNAPTIC WEIGHT

  Spt = zeros(Float64, (NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2))
  deltaWpt = zeros(Float64, (NTdim1 + 2, NTdim2 + 2, NRdim1 + 2, NRdim2 + 2))
  totalSp = zeros(Float64, (NRdim1 + 2, NRdim2 + 2))
  meanSp = zeros(Float64, (NRdim1 + 2, NRdim2 + 2))
  deltaWsum = zeros(Float64, (NRdim1 + 2, NRdim2 + 2))
  connections = zeros(Int, (NRdim1 + 2, NRdim2 + 2))

  for rdim1 in rmindim1 + pad : rmaxdim1 + pad
    for rdim2 in rmindim2 + pad : rmaxdim2 + pad
      for tdim1 in tmindim1 : tmaxdim1 + pad
        for tdim2 in tmindim2 : tmaxdim2 + pad

          # Calculate similarity
          Spt[tdim1, tdim2, rdim1, rdim2] = sum(min(normalisedCpm[:, rdim1, rdim2], normalisedCtm[:, tdim1, tdim2]))

          # Count connections
          if Wpt[currentiteration - 1, tdim1, tdim2, rdim1, rdim2] > 0.
            totalSp[rdim1, rdim2] += Spt[tdim1, tdim2, rdim1, rdim2]
            connections[rdim1, rdim2] += 1
          end
        end
      end
    end
  end
  # Calculate mean similarity
  meanSp[rmindim1+pad:rmaxdim1 + pad, rmindim2+pad:rmaxdim2 + pad] = (totalSp[rmindim1+pad:rmaxdim1 + pad,rmindim2+pad:rmaxdim2 + pad] ./ connections[rmindim1+pad:rmaxdim1 + pad,rmindim2+pad:rmaxdim2 + pad]) - k


  for rdim1 in rmindim1 + pad : rmaxdim1 + pad
    for rdim2 in rmindim2 + pad : rmaxdim2 + pad
      # Calculate deltaW
      deltaWpt[tmindim1:tmaxdim1 + 1, tmindim2:tmaxdim2 + 1, rdim1, rdim2] = h * (
      Spt[tmindim1:tmaxdim1 + 1, tmindim2:tmaxdim2 + 1, rdim1, rdim2] - meanSp[rdim1, rdim2])

      for tdim1 in tmindim1 + pad : tmaxdim1 + pad
        for tdim2 in tmindim2 + pad : tmaxdim2 + pad

          # Calculate deltaWsum
          if Wpt[currentiteration - 1, tdim1, tdim2, rdim1, rdim2] > 0.
            deltaWsum[rdim1, rdim2] += deltaWpt[tdim1, tdim2, rdim1, rdim2]
          end
        end
      end

      # Update Weight
      Wpt[currentiteration, :, :, rdim1, rdim2] = (Wpt[currentiteration - 1, :, :, rdim1, rdim2] + deltaWpt[:, :,
      rdim1,
      rdim2]) / (
      W + deltaWsum[rdim1, rdim2])
    end
  end
end


function removesynapses()
  for tdim1 in tmindim1 + pad : tmaxdim1 + pad
    for tdim2 in tmindim2 + pad : tmaxdim2 + pad
      for rdim1 in rmindim1 + pad : rmaxdim1 + pad
        for rdim2 in rmindim2 + pad : rmaxdim2 + pad
          if Wpt[currentiteration, tdim1, tdim2, rdim1, rdim2] < elim * W
            Wpt[currentiteration, tdim1, tdim2, rdim1, rdim2] = 0.
          end
        end
      end
    end
  end
end


function addsynapses()
  for tdim1 in tmindim1 + pad : tmaxdim1 + pad
    for tdim2 in tmindim2 + pad : tmaxdim2 + pad
      for rdim1 in rmindim1 + pad : rmaxdim1 + pad
        for rdim2 in rmindim2 + pad : rmaxdim2 + pad
          if Wpt[currentiteration, tdim1, tdim2, rdim1, rdim2] == 0. && (
            Wpt[currentiteration, tdim1 + 1, tdim2, rdim1, rdim2] > 0.02 * W || Wpt[
            currentiteration,
            tdim1 - 1, tdim2, rdim1, rdim2] > 0.02 * W || Wpt[currentiteration,
            tdim1, tdim2 + 1, rdim1, rdim2] > 0.02 * W ||
            Wpt[currentiteration,
            tdim1, tdim2 - 1, rdim1, rdim2] > 0.02 * W)
            Wpt[currentiteration, tdim1, tdim2, rdim1, rdim2] = newW * W
          end
        end
      end
    end
  end
end




######################## ALGORITHM ##########################


# MARKER LOCATIONS

if Mdim1 > 1
  markerspacingdim1 = NRdim1 / (Mdim1 - 1)
else
  markerspacingdim1 = 0
end
if Mdim2 > 1
  markerspacingdim2 = NRdim2 / (Mdim2 - 1)
else
  markerspacingdim2 = 0
end

m = 1
locationdim1 = 1
locationdim2 = 1
for mdim2 in 1:Mdim2 - 1
  for mdim1 in 1:Mdim1 - 1
    Qpm[m, round(Int, locationdim1) + pad, round(Int, locationdim2) + pad] = Q
    locationdim1 += markerspacingdim1
    m += 1
  end
  Qpm[m, NRdim1, locationdim2] = Q
  locationdim1 = 1
  locationdim2 += markerspacingdim2
  m += 1
end

for mdim1 in 1:Mdim1 - 1
  Qpm[m, round(Int, locationdim1), NRdim2] = Q
  locationdim1 += markerspacingdim1
  m += 1
end
Qpm[m, NRdim1, NRdim2] = Q

# PRESYNAPTIC CONCENTRATIONS

updateNc()
averagemarkerchange = 1
while averagemarkerchange > stab
  deltaconc = conc_change(Cpm, "presynaptic")
  averagemarkerchange = (sum(sum(sum(deltaconc))) / sum(sum(sum(Cpm)))) * 100
  Cpm += (deltaconc * deltat)
end
normalisedCpm = normalise(Cpm, "presynaptic")

# INITIAL CONNECTIONS

for rdim1 in rmindim1 + pad : rmaxdim1 + pad
  for rdim2 in rmindim2 + pad : rmaxdim2 + pad
    initialconections(rdim1, rdim2)
  end
end

# INITIAL CONCENTRATIONS

updateQtm()
for t in 1:td
  deltaconc = conc_change(Ctm, "tectal")
  Ctm += (deltaconc * deltat)
end
normalisedCtm = normalise(Ctm, "tectal")

# ITERATIONS

for iteration in 1 : Iterations
  currentiteration += 1
  updateWeight()
  removesynapses()
  addsynapses()

  updateQtm()
  for t in 1:td
    deltaconc = conc_change(Ctm, "tectal")
    Ctm += (deltaconc * deltat)
  end
  normalisedCtm = normalise(Ctm, "tectal")
end



timeelapsed = time() - starttime
print("Time elapsed: ", timeelapsed, " seconds")
