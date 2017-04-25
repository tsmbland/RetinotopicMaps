# RetinotopicMaps

This repository contains Python code for modelling and analysing the marker induction mechanism of retinotopic map formation. Code is ready to use, and should be run using Python 3.4.0 or later. Data produced from simulations is saved in a separate folder called ‘RetinotopicMapsData’, which will be automatically created if not available.

A typical full simulation will involve the following steps, in order:
1)	Set model parameters
2)	Run a simulation to generate a weight matrix
3)	Analyse this weight matrix to characterise receptive fields
4)	Analyse receptive fields to quantify map precision
5)	Plot data

The repository contains 6 folders containing scripts used to carry out these tasks, explained in detail below:


## Models

Contains scripts for simulating three versions of the marker induction model: the original 1979 model, the updated 2006 model, and the hybrid model developed in this project. Each folder contains a number of files:

* Parameters.py: used to set model parameters‘Functions’: contains functions for all calculations in the model
* StandardAlgorithm.py: normal development with a fixed-size retina and tectum and standard initial gradients
* NoSurgery.py: imports data generated from a normal simulation (user specified) and continues map development as normal
* MismatchSurgery.py: concentration profiles must be imported from a fully developed normal map (user specified). Script then simulates surgery operations (as specified in the parameters file) and subsequent map development.
* EphA3Knockin.py: map development with a retinal EphA3 knockin profile
* Development.py: development with a growing retina and tectum. NB Not fully tested, and not presented in the project report.

In order to run a simulation choose a model, specify parameters in the ‘Parameters’ file, and then run the necessary algorithm script. Algorithm scripts contain support for multiprocessing (e.g. can run 10 simultaneous simulations with 10 different values of a certain parameter). Matrices of data (synaptic weights, receptor and ligand densities etc.) are saved in the folder ‘RetinotopicMapsData/JOBID’, where JOBID is a unique simulation-identifying code, specified by the user in the Parameters file. Data is saved over the course of the simulation with a time resolution specified in the Parameters file.


## ReceptiveFields / ProjectiveFields

Contains scripts for characterising receptive/projective fields of tectal cells/RGCs. Imports previously generated synaptic weight data, and exports data into the same directory. Multiprocessing feature allows batches of jobs to be performed simultaneously. To perform analysis run the ‘Algorithm’ file, and specify JOBIDs when prompted.


## MapPrecision

Contains scripts for calculating precision measures. Imports previously generated receptive field data, and exports precision data to the same directory. Calculates measures both excluding and including boundary regions (width of excluded region is specified in the Algorithm file). Multiprocessing feature allows batches of jobs to be performed simultaneously. To perform analysis run the ‘Algorithm’ file, and specify JOBIDs when prompted. NB Currently only supports precision measures based on receptive fields (and not projective fields), but could be easily adapted if required. 


## Plots

Contains scripts for generating a standard set of plots from simulation data. To display plots run the relevant script and enter the JOBID when prompted. Some plots have further options within the script file.

* AreaPlot.py¹: plots the connections and receptive field estimate for a given RGC
* ConcPlot.py¹²: plots tectal marker concentrations (1979 model)
* EphA3Plot.py³: plots retinal Eph gradients in EphA3 knockin simulations
* EphA3ProjectionPlot.py¹: creates lattice plots of the retina onto the tectum for EphA3 knockin simulations
* EphPlot.py³: plots retinal Eph gradients in wild type simulations
* EphrinPlot.py¹³: plots tectal ephrin gradients
* FieldPlot.py¹: creates a lattice plot of the tectum onto the retina
* PrecisionPlot.py: plots the value of map precision measures over development time
* ProjectionPlot.py¹: creates a lattice plot of the retina onto the tectum
* WeightPlot.py¹: 2D plot of the connections between a slice of RGCs and a slice of tectal cells

¹ Plots are animated to allow data to be followed over time
² Compatible only with the 1979 version of the model
³ Compatible only with the 2006 and hybrid versions of the model


## ReportFigs

Contains scripts for figures used in my report.
