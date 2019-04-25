import sys, glob, os, time
import numpy as np

from mpi4py import MPI
from MedGA_sequential import MedGA

WORKTAG = 0
DIETAG = 1

# Master process. It distributes the images among the slaves and collects the elapsed times
def master(toProcess, pathsOutput, population, generations, selection, cross_rate, mut_rate, elitism, pressure, verbose):

	n = len(toProcess)
	status = MPI.Status()

	times = np.zeros(len(toProcess))
	idx = 0 

	# If the number of images (n) is greater than the available Slaves (size-1)
	# (size-1) images are run in parallel
	if n > (size-1):
		for i in range(1, size):
			inp = [toProcess[i-1], pathsOutput[i-1], population, generations, selection, cross_rate, mut_rate, elitism, pressure, verbose]
			comm.send(inp, dest=i, tag=WORKTAG)

		# As soon as a Slave is available, the master assigns it a new image to process
		for i in range(size, n+1):
			im_free, elapsed = comm.recv(source=MPI.ANY_SOURCE, tag=10, status=status)
			times[idx] = elapsed
			idx += 1

			inp = [toProcess[i-1], pathsOutput[i-1], population, generations, selection, cross_rate, mut_rate, elitism, pressure, verbose]
			comm.send(inp, dest=im_free, tag=WORKTAG)
	
	# If the number of images (n) is lower than the available cores (size-1)
	# only n Slaves are used
	else:
		for i in range(0, n):
			inp = [toProcess[i], pathsOutput[i], population, generations, selection, cross_rate, mut_rate, elitism, pressure, verbose]
			comm.send(inp, dest=i+1, tag=WORKTAG)

		for i in range(0, n):
			im_free, elapsed = comm.recv(source=MPI.ANY_SOURCE, tag=10, status=status)
			times[idx] = elapsed
			idx += 1

	# The Master sends the DIETAG to the Slaves
	for i in range(1, size):
		comm.send(obj=None, dest=i, tag=DIETAG)

	return times

def slave():

	while True:
		status = MPI.Status()

		# The Slave waits an image to process
		inp = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
		elapsed = 0
		
		if inp != None:

			start = time.time()

			# MedGA execution on the provided image by using the provided GA settings
			medga = MedGA(inp[0], inp[1])
			medga.startGA(inp[2], inp[3], inp[4], inp[5], inp[6], inp[7], inp[8])

			end = time.time()
			elapsed = end-start

			if inp[9]:
				sys.stdout.write(" * Analyzed image %s"%inp[0])
				sys.stdout.write(" -> Elapsed time %5.2fs on rank %d\n\n" % (elapsed, rank))

		if status.Get_tag():
			break
		
		# The Slave is free to process a new image
		comm.send([rank,elapsed], dest=0, tag=10)

if __name__ == '__main__':

	folderIn   = sys.argv[1]
	folderOut  = sys.argv[2]

	population  = int(sys.argv[3])
	generations = int(sys.argv[4])
	selection   = sys.argv[5]
	cross_rate  = float(sys.argv[6])
	mut_rate    = float(sys.argv[7])
	elitism     = int(sys.argv[8])
	pressure    = int(sys.argv[9])
	verbose     = bool(sys.argv[10])

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	comm.Barrier()

	# Master process
	if rank == 0:
		sys.stdout.write(" * MedGA is using %d cores\n\n\n" % (size))

		startAll = time.time()

		toProcess   = []
		pathsOutput = []

		# Looking for the images in the provided input folder
		listImages = glob.glob(folderIn+os.sep+"*")
		for imagePath in listImages:
			ext = imagePath.split(".")[-1].lower()
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

				# Output folders
				string    = imagePath.split("/")[1:]
				subfolder = string[-1].split(".")[:-1]

				pathOutput = folderOut+os.sep+subfolder[0]

				if not os.path.exists(pathOutput):
					os.makedirs(pathOutput)

				pathsOutput.append(pathOutput)

		times = master(toProcess, pathsOutput, population, generations, selection, cross_rate, mut_rate, elitism, pressure, verbose)

	# Slave process
	else:
		slave()

	comm.Barrier()
	if rank == 0:
		endAll     = time.time()
		elapsedAll = endAll-startAll

		if verbose:
			if len(toProcess) > 1:
				sys.stdout.write("\n * Total elapsed time %5.2fs for computing %d images\n" % (elapsedAll, len(toProcess)))
				sys.stdout.write(" * Mean elapsed time  %5.2fs per image\n" % (np.mean(times)))
			sys.stdout.write('********************************************************************************\n')

