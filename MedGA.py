"""
 * L. Rundo, A. Tangherloni et al.: MedGA: a novel evolutionary method for image enhancement in medical imaging systems,
   Expert Systems with Applications, 119, 387-399, 2019. doi: 10.1016/j.eswa.2018.11.013

 * L. Rundo, A. Tangherloni et al.: A novel framework for MR image segmentation and quantification by using MedGA,
   Computer Methods and Programs in Biomedicine, 2019. doi: 10.1016/j.cmpb.2019.04.016

 
 * Copyright (C) 2019 - Andrea Tangherloni & Leonardo Rundo
 * Distributed under the terms of the GNU General Public License (GPL)
 * This file is part of MedGA.

 * MedGA is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License v3.0 as published by
 * the Free Software Foundation.
  
 * MedGA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the
 * GNU General Public License for more details.
"""

import getopt, sys, glob, os, time, subprocess
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from MedGA_sequential import MedGA

# MPI version of MedGA. It requires both MPI and mpi4py
def runMPI(folderIn, folderOut, population, generations, selection, cross_rate, mut_rate, pressure, elitism, cores, verbose):
	if verbose:
		print '********************************************************************************'
		print "* Running the MPI version of MedGA\n"

		print " * GA settings"
		print "   -> Number of chromosome: %d"%population
		print "   -> Number of elite chromosomes: %d"%elitism
		print "   -> Number of generations: %d"%generations
		print "   -> Crossover rate: %.2f"%cross_rate
		print "   -> Mutation rate:  %.2f"%mut_rate

		if selection == 'wheel':
			print "   -> Selection: wheel roulette\n\n"
		elif selection == 'ranking':
			print "   -> Selection: ranking \n\n"
		else:
			print "   -> Selection: tournament with %d individuals\n"%pressure

	if not os.path.exists(folderOut):
		os.makedirs(folderOut)

	try:
		# Using mpiexec
		run = "mpiexec -np %d python src/MedGA_mpi.py %s %s %d %d %s %f %f %d %d %s"%(cores, folderIn, folderOut,
			                                                                          population, generations, selection,
			                                                                          cross_rate, mut_rate, pressure,
			                                                                          elitism, str(verbose))
		
		# Calling the MPI version of MedGA, which is based on the sequential version
		p = subprocess.call(run, shell=True)
	except:
		# Using mpirun
		run = "mpirun -np %d python src/MedGA_mpi.py %s %s %d %d %s %f %f %d %d %s"%(cores, folderIn, folderOut,
			                                                                         population, generations, selection,
			                                                                         cross_rate, mut_rate, pressure,
			                                                                         elitism, str(verbose))
		
		# Calling the MPI version of MedGA, which is based on the sequential version
		p = subprocess.call(run, shell=True)

# Sequential version of MedGA
def run(imagePath, folderIn, folderOut, population, generations, selection, cross_rate, mut_rate, pressure, elitism, verbose):
	
	if verbose:
		print '********************************************************************************'
		print "* Running the sequential version of MedGA\n"

		print " * GA settings"
		print "   -> Number of chromosome: %d"%population
		print "   -> Number of elite chromosomes: %d"%elitism
		print "   -> Number of generations: %d"%generations
		print "   -> Crossover rate: %.2f"%cross_rate
		print "   -> Mutation rate:  %.2f"%mut_rate

		if selection == 'wheel':
			print "   -> Selection: wheel roulette\n\n"
		elif selection == 'ranking':
			print "   -> Selection: ranking \n\n"
		else:
			print "   -> Selection: tournament with %d individuals\n\n"%pressure

	startAll = time.time()

	toProcess = []

	# Looking for the provided input image
	if folderIn is None:

		ext = imagePath.split(".")[-1].lower()
		listExts = ["tiff", "tif", "png", "jpeg", "jpg"]

		if ext not in listExts:
			print "Unsupported format. Please provide", listExts, "images"
			print imagePath, "has not been processed"
			print "Warning", imagePath, "has not been processed"
			print '********************************************************************************'
			exit(-6)

		if not os.path.exists(imagePath):
			print imagePath, "does not exists"
			print "Warning", imagePath, "has not been processed"
			print '********************************************************************************'
			exit(-7)

		else:
			toProcess.append(imagePath)


	else:
		# Looking for the images in the provided input folder
		listImages = glob.glob(folderIn+os.sep+"*")
		for imagePath in listImages:
			ext = imagePath.split(".")[-1]
			listExts = ["tiff", "tif", "png", "png", "jpeg", "jpg"]

			# Only tiff, png and jpg images can be elaborated
			if ext not in listExts:
				print "Unsupported format. Please provide", listExts, "images"
				print "Warning", imagePath, "will be not processed"
				pass

			if not os.path.exists(imagePath):
				print imagePath, "does not exists"
				print "Warning", imagePath, "will be not processed"
			
			else:
				toProcess.append(imagePath)

	if not os.path.exists(folderOut):
		os.makedirs(folderOut)

	times = np.zeros(len(toProcess))
	
	# Processing the images in the input folder
	for i in xrange(len(toProcess)):

		# Output folders
		string    = toProcess[i].split("/")[1:]
		subfolder = string[-1].split(".")[:-1]

		pathOutput = folderOut+os.sep+subfolder[0]

		if not os.path.exists(pathOutput):
			os.makedirs(pathOutput)

		if verbose:
			print " * Analyzed image %s"%toProcess[i], 

		start = time.time()
		
		# MedGA execution on the image to be processed by using the provided GA settings
		medga = MedGA(toProcess[i], pathOutput)
		medga.startGA(population, generations, selection, cross_rate, mut_rate, elitism, pressure)

		end = time.time()
		elapsed = end-start
		times[i] = elapsed

		if verbose:
			print "-> Elapsed time %5.2fs" % (elapsed)
			print

		endAll     = time.time()
		elapsedAll = endAll-startAll

	if verbose:
		if len(toProcess) > 1:
			print "\n * Total elapsed time %5.2fs" % elapsedAll, "for computing", len(toProcess), "images"
			print " * Mean elapsed time  %5.2fs per image" % np.mean(times)
		print '********************************************************************************'

if __name__ == '__main__':

	helpString = """MedGA.py -h <help>
         -i <image>
         -f <folder>
         -o <output>      (default: output)
         -p <population>  (default: 100)
         -g <generations> (default: 100)
         -s <selection>   (default: tournament)
         -c <cross_rate>  (default: 0.9)
         -m <mut_rate>    (default: 0.01)
         -k <pressure>    (default: 20)
         -e <elitism>     (default: 1)
         -d <distributed> (default: False)
         -t <cores>       (default: 4)
         -v <verbose>     (default: False)"""
	
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hi:f:o:p:g:s:c:m:k:e:t:dv", ["help", "image", "folder", "output",
			                                                                   "population", "generations", "selection",
			                                                                   "cross_rate", "mut_rate", "pressure",
			                                                                   "elitism", "cores", "distributed", "verbose"])
	except:
		print helpString
		exit(-1)

	if len(opts)==0:
		print helpString
		exit(-2)

	# default settings of MedGA
	imagePath   = None
	folderIn    = None
	folderOut   = "output"
	population  = 100
	generations = 100
	selection   = "tournament"
	cross_rate  = 0.9
	mut_rate    = 0.01
	pressure    = 20
	elitism     = 1
	mpi         = False
	cores       = 4
	verbose     = False

	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print helpString
			exit(-3)
		elif opt in ("-i", "--image"):
			imagePath = arg
		elif opt in ("-f", "--folder"):
			folderIn = arg
		elif opt in ("-o", "--output"):
			folderOut = arg
		elif opt in ("-p", "--population"):
			population = int(arg)
		elif opt in ("-g", "--generations"):
			generations = int(arg)
		elif opt in ("-s", "--selection"):
			selection = arg
		elif opt in ("-c", "--cross_rate"):
			cross_rate = float(arg)
		elif opt in ("-m", "--mut_rate"):
			mut_rate = float(arg)
		elif opt in ("-k", "--pressure"):
			pressure = int(arg)
		elif opt in ("-e", "--elitism"):
			elitism = int(arg)
		elif opt in ("-d", "--distributed"):
			mpi = True
		elif opt in ("-t", "--cores"):
			cores = int(arg)
		elif opt in ("-v", "--verbose"):
			verbose = True


# ************************************************ Checking provided settings ************************************************
	warning = False
	if population <= 0:
		print " * Warning, the provided population is %d. It has been set to 100"%population
		cross_rate = 100
		warning = True

	if (cross_rate < 0) or (cross_rate > 1):
		print " * Warning, the provided cross_rate is %f. It has been set to 0.9"%cross_rate
		cross_rate = 0.9
		warning = True

	if (mut_rate < 0) or (mut_rate > 1):
		print " * Warning, the provided mut_rate is %f. It has been set to 0.01"%mut_rate
		mut_rate = 0.01
		warning = True

	if (pressure > population) or (pressure <= 0):
		print " * Warning, the provided pressure is %d. It has been set to 20"%pressure
		pressure = 20
		warning = True

	if (elitism > population) or (elitism < 0):
		print " * Warning, the provided elitism is %d. It has been set to 1"%elitism
		elitism = 1
		warning = True

	if selection not in ["wheel", "ranking", "tournament"]:
		print " * Warning, the provided selection is %s. It has been set to tournament"%selection
		selection = "tournament"
		warning = True

	if cores <= 1:
		print " * Warning, the provided number of cores is %d. It has been set to 4"%cores
		cores = 4
		warning = True

	if (imagePath is None) and (folderIn is None):
		print " * Please, provide either an image or a folder containing at least an image"
		exit(-4)

	if (imagePath is not None) and (folderIn is not None):
		print " * Please, provide either an image or a folder containing at least an image"
		exit(-5)

	if warning:
		print


# ************************************************ Running MedGA ************************************************
	if mpi and folderIn is not None:
		# Run MPI version on a folder
		runMPI(folderIn, folderOut, population, generations, selection, cross_rate, mut_rate, pressure, elitism, cores, verbose)
	else:
		# Run sequential version on either a folder or a single image
		run(imagePath, folderIn, folderOut, population, generations, selection, cross_rate, mut_rate, pressure, elitism, verbose)
