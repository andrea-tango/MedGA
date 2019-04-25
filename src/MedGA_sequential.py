import os
import random as rnd
import numpy as np
from copy import deepcopy
import sys
import time

from GA import chromosome
from GA import geneticOperation
from imageProcessing import processing

# Class containing MedGA
class MedGA(object):

	def __init__(self, pathInput, pathOutput):

		self.__pathIn	= pathInput
		self.__pathOut	= pathOutput

		self.__outputName 	 	= None
		self.__outputNameFit 	= None
		self.__outputNameThresh	= None
		self.__outputNameTerms	= None
		self.__outputNameInfo	= None

		self.__childrenPerGen 	= None
		self.__numberOfGenes	= None
		self.__minGrayLevel		= None
		self.__maxGrayLevel		= None
		self.__targetMatrix		= None
		self.__targetHist		= None
		self.__noZeroPosHist	= None


	def startGA(self, pop_size, numGen, selection, cross_rate, mut_rate, elitism, numberIndTour, minGL = 1):

		#Image Processing object
		imPros = processing()
		
		# Paths used to save the images and other information
		self.__outputNameInfo = self.__pathOut + os.sep + 'information'
		
		self.__outputName 	  = self.__pathOut + os.sep + 'images'
		
		if not os.path.exists(self.__outputName):
			os.makedirs(self.__outputName)

		self.__outputNameFit	= self.__pathOut + os.sep + "fitness"
		self.__outputNameThresh	= self.__pathOut + os.sep + "threshold"
		self.__outputNameTerms	= self.__pathOut + os.sep + "terms"

		# Reading the input image
		self.__targetMatrix, numberGrayLevel, self.__targetHist, self.__noZeroPosHist, maxValueGray, T_k  = imPros.loadImage(self.__pathIn, self.__outputName )

		# GAs settings
		self.__childrenPerGen 	= pop_size - elitism # number of new individuals for each generation
		self.__numberOfGenes	= numberGrayLevel
		self.__minGrayLevel		= minGL
		self.__maxGrayLevel		= maxValueGray

		# Saving the used GA settings
		with open(self.__outputNameInfo, "w") as fo:
			fo.write("******************************************************\n")
			fo.write("\t\t\t GA settings\n\n")
			fo.write("Number of chromosome: " + str(pop_size) + "\n")
			fo.write("Number of elite chromosomes: " + str(elitism) + "\n")
			fo.write("Number of genes: " + str(self.__numberOfGenes) + "\n")
			fo.write("Number of generations: " + str(numGen) + "\n")
			fo.write("Crossover rate: " + str(cross_rate) + "\n")
			fo.write("Mutation rate:  " + str(mut_rate) + "\n")
			
		pop = []

		# Initialization of the GA instance
		pop = self.__initialize(pop, pop_size, mut_rate, T_k)

		# Evolution of the GA 
		pop = self.__evolve(pop, cross_rate, mut_rate, numGen, elitism, T_k, method=selection, numberInd = numberIndTour)


	# Initializing the population
	def __initialize(self, pop, pop_size, mut_rate, T_k):

		for i in xrange(pop_size):
			pop.append(chromosome(self.__targetHist, self.__noZeroPosHist, self.__numberOfGenes, self.__minGrayLevel, self.__maxGrayLevel, mut_rate, T_k))

		# Sorting the population based on the fitness values
		pop.sort(key=lambda x: x.getFitness())

		pop[0].saveCurrentImage(self.__targetHist, self.__noZeroPosHist, self.__targetMatrix, self.__outputName + os.sep + 'image0.png', self.__outputName + os.sep+ 'imageConf0.png')

		with open(self.__outputNameFit , "w") as fo:
			fo.write(str(pop[0].getFitness()) + "\n")

		with open(self.__outputNameThresh , "w") as fo:
			fo.write(str(pop[0].getOpt_T()) + "\n")

		pop[0].saveTermFitness(self.__outputNameTerms, "w")

		return pop

	# Evolution of the population
	def __evolve(self, pop, cross_rate, mut_rate, numGen, elitism, T_k, method = 'wheel', numberInd = 10):

		n = len(pop)
		with open(self.__outputNameInfo, "a") as fo:
			if method == 'wheel':
				fo.write("Selection: wheel roulette\n")
			elif method == 'ranking':
				fo.write("Selection: ranking \n")
			else:
				fo.write("Selection: tournament with " + str(numberInd) + " individuals\n")

		op = geneticOperation()
		
		# The population evolves for numGen-1 generations
		for i in range(1, numGen):

			# Roulette wheel selection
			if method == 'wheel':
				probabilities = []
				for j in range(0, n):
					probabilities.append(pop[j].getFitness())
				sum_fit = np.sum(probabilities)

				for j in range(0, n):
					probabilities[j] = (probabilities[j]) / (sum_fit * 1.0)

				probabilities = (1 - np.array(probabilities))
				probabilities /= np.sum(probabilities)

			# Ranking selection
			elif method == 'ranking':
				probabilities = []
				rank = np.linspace(1, n, n)
				rank = rank[::-1]
				probabilities = rank / float(np.sum(rank))

			# New individuals
			countWhile = self.__childrenPerGen
			children = []

			while(countWhile > 0):

				# Tournament selection
				if method == 'tournament':
					dist1 = np.random.randint(0, n, numberInd)
					dist2 = np.random.randint(0, n, numberInd)

					individuals1 = []
					individuals2 = []
					for k in xrange(numberInd):
						individuals1.append(pop[dist1[k]])
						individuals2.append(pop[dist2[k]])
	
					individuals1.sort(key=lambda x: x.getFitness())
					individuals2.sort(key=lambda x: x.getFitness())

					parent_1 = individuals1[0]
					parent_2 = individuals2[0]

				# Roulette wheel or ranking selection
				else:
					parent_1 = np.random.choice(pop, p=probabilities)
					parent_2 = np.random.choice(pop, p=probabilities)
				
				# The latest individual is the best children
				if (countWhile == 1):
					child0 = chromosome(self.__targetHist, self.__noZeroPosHist, self.__numberOfGenes, self.__minGrayLevel, self.__maxGrayLevel, mut_rate, T_k, parent_1=parent_1, parent_2=parent_2)
					child1 = deepcopy(chromosome(self.__targetHist, self.__noZeroPosHist, self.__numberOfGenes, self.__minGrayLevel, self.__maxGrayLevel, mut_rate, T_k, parent_1=parent_1, parent_2=parent_2, cross_point=child0.crossPoint))

					if child0.getFitness() < child1.getFitness():
						children.append(child0)
					else:
						children.append(child1)
						countWhile = countWhile - 1

				# Crossover
				elif (rnd.uniform(0, 1) < cross_rate):
					child0 = chromosome(self.__targetHist, self.__noZeroPosHist, self.__numberOfGenes, self.__minGrayLevel, self.__maxGrayLevel, mut_rate, T_k, parent_1=parent_1, parent_2=parent_2)
					children.append(child0)
					child1 = deepcopy(chromosome(self.__targetHist, self.__noZeroPosHist, self.__numberOfGenes, self.__minGrayLevel, self.__maxGrayLevel, mut_rate, T_k, parent_1=parent_1, parent_2=parent_2, cross_point=child0.crossPoint))
					children.append(child1)
					countWhile = countWhile - 2

				# Parents without crossover
				else:
					op.mutate(parent_1.genes, self.__minGrayLevel, self.__maxGrayLevel, parent_1.getOpt_T(), mut_rate)
					op.mutate(parent_2.genes, self.__minGrayLevel, self.__maxGrayLevel, parent_2.getOpt_T(), mut_rate)
					parent_1.calculateFitness(self.__targetHist, self.__noZeroPosHist, self.__maxGrayLevel, self.__minGrayLevel)
					parent_2.calculateFitness(self.__targetHist, self.__noZeroPosHist, self.__maxGrayLevel, self.__minGrayLevel)
					children.append(parent_1)
					children.append(parent_2)
					countWhile = countWhile-2

			# Elitism to mantain the best individual(s) during the evolution
			pop[elitism:n] = deepcopy(children[0:self.__childrenPerGen])

			# Sorting the population based on the fitness values
			pop.sort(key=lambda x: x.getFitness())

			with open(self.__outputNameFit , "a") as fo:
				fo.write(str(pop[0].getFitness()) + "\n")

			with open(self.__outputNameThresh , "a") as fo:
				fo.write(str(pop[0].getOpt_T()) + "\n")

			pop[0].saveTermFitness(self.__outputNameTerms, "a")

			if i == numGen - 1:
				pop[0].saveCurrentImage(self.__targetHist, self.__noZeroPosHist, self.__targetMatrix, self.__outputName + os.sep + 'imageBest.png', self.__outputName + os.sep + 'imageConfBest.png')

				np.savetxt(self.__pathOut + os.sep + 'matrixBest', pop[0].getMatrix(), fmt='%d')
		return pop