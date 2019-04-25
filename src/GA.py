import random as rnd
import numpy as np
import math
import scipy.misc
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Class containing the chromosome (individual) structure
class chromosome(object):
    
    def __init__(self, targetHist, noZeroPosHist, numberOfGenes, minGrayLevel, maxGrayLevel, mut_rate, T_k, parent_1=None, parent_2=None, cross_point=None):
        
        self.genes      = []
        self.crossPoint = None
        self.__fitness  = None
        self.__opt_T    = 0
        self.__hist     = None
        self.__matrix   = None
        self.__term1    = None
        self.__term2    = None
        self.__term3    = None

        if parent_1 and parent_2:
            op = geneticOperation()
            self.genes, self.crossPoint = op.crossoverUniform(parent_1, parent_2, cross_point)
            op.mutate(self.genes, minGrayLevel, maxGrayLevel, self.__opt_T, mut_rate)

        else:

            dist = self.__generateUniformDistribution(noZeroPosHist, minGrayLevel, maxGrayLevel)

            for i in range(0, numberOfGenes):
                self.genes.append(gene(dist[i]))
        
        self.genes.sort(key=lambda x: x.position)
        self.__fitness, self.__opt_T, self.__hist = self.calculateFitness(targetHist, noZeroPosHist, maxGrayLevel, minGrayLevel)

    def __generateUniformDistribution(self, noZeroPosHist, minGrayLevel, maxGrayLevel):

        dist1 =  np.random.uniform(minGrayLevel, maxGrayLevel, len(noZeroPosHist))
        dist = [int(round(j)) for j in dist1]
        return sorted(dist)

    def __calculateVariances(self, hist, opt_T, mu1, mu2):

        noZeroPosHist = np.nonzero(hist)[0]

        val1 = noZeroPosHist[0]
        val2 = noZeroPosHist[0]
        pos = 0
        for i in range(1, len(noZeroPosHist)):
            if noZeroPosHist[i] <= opt_T:
                val2 = noZeroPosHist[i]
                pos = i
            else:
                break

        val3 = noZeroPosHist[pos+1]
        val4 = noZeroPosHist[-1]

        halfWidth1 = (val2 - val1) / 2.0
        halfWidth2 = (val4 - val3) / 2.0

        countOcc1 = 0
        acc1 = 0
        countOcc2 = 0
        acc2 = 0
        for i in xrange(len(noZeroPosHist)):
            greyLev = noZeroPosHist[i]
            if greyLev <= opt_T:
                countOcc1 += hist[greyLev]
                acc1 += (hist[greyLev] * (greyLev - mu1)**2)
            else:
                countOcc2 += hist[greyLev]
                acc2 += (hist[greyLev] * (greyLev - mu2)**2)                

        std1 = math.sqrt((acc1 / (float(countOcc1))))
        std2 = math.sqrt((acc2 / (float(countOcc2))))

        return std1, std2, halfWidth1, halfWidth2

    def __optimalThreshold(self, hist_vals, delta_T, max_it):

        #Initializing the output value
        opt_T = 1
    
        #Initializing the threshold to the global mean
        h_dim = len(hist_vals)
        total_pixel_number = np.sum(hist_vals)
        weighted_hist_sum = 0
        for i in range(0, h_dim):
            weighted_hist_sum = weighted_hist_sum + hist_vals[i] * (i-1)
        
        hist_mean = weighted_hist_sum / (total_pixel_number*1.0)
        
        #If the the histogram mean is equal to 0, the procedure ends  
        if hist_mean == 0:
            return opt_T
    
        #Threshold at step k
        T_k = 0
        
        #Threshold at step k+1
        T_k1 = int(math.floor(hist_mean))
        
        # Iteration counter
        counter = 1
        
        while counter < max_it:
            if (T_k1 - T_k) <= delta_T:
                break
            
            # Updating the threshold
            T_k = T_k1
            
            # Splitting the histogram H in two sub-histograms H1 and H2 by means of the threshold T_k
            H1_pixel_number = 0
            H2_pixel_number = 0
            for i in range(0, T_k):
                H1_pixel_number += hist_vals[i]
            for i in range((T_k+1), h_dim):
                H2_pixel_number += hist_vals[i]
                            
            weighted_H1_sum = 0
            for i in range(0, T_k):
                weighted_H1_sum = weighted_H1_sum + hist_vals[i] * (i-1)
            
            weighted_H2_sum = 0
            for i in range((T_k+1), h_dim):
                weighted_H2_sum = weighted_H2_sum + hist_vals[i] * (i-1)
            
            # Mean value of the sub-histogram H1
            H1_mean = weighted_H1_sum / (H1_pixel_number*1.0)
            
            # Mean value of the sub-histogram H2
            H2_mean = weighted_H2_sum / (H2_pixel_number*1.0)
            
            # Updating the threshold at step k+1 (T_k1)
            T_k1 = int(math.floor((H1_mean + H2_mean) / 2.0))
            counter = counter + 1
            
        return T_k, H1_mean, H2_mean

    def saveCurrentImage(self, targetHist, noZeroPosHist, targetMatrix, f_name, f_nameConf):

        self.__matrix = deepcopy(targetMatrix)
        newNoZeros = []
        for i in range(0, len(self.genes)):
            newNoZeros.append(self.genes[i].position)

        for i in range(0, len(noZeroPosHist)):
            ind = noZeroPosHist[i]
            pos = np.where(targetMatrix == ind);
            
            pos_x = pos[0]
            pos_y = pos[1]
            for j in range(0, len(pos_x)):
               self.__matrix[pos_x[j]][pos_y[j]] = newNoZeros[i]

        plt.figure()
        plt.subplot(121)
        plt.imshow(targetMatrix, cmap='Greys_r')
        plt.subplot(122)
        plt.imshow(self.__matrix, cmap='Greys_r')
        plt.tight_layout()
        plt.savefig(f_nameConf)
        plt.close()

        scipy.misc.imsave(f_name, self.__matrix)

    def saveTermFitness(self, file, mod):
        with open(file, mod) as fo:
            fo.write(str(self.__term1) + "\t")
            fo.write(str(self.__term2) + "\t")
            fo.write(str(self.__term3) + "\n")

    def calculateFitness(self, targetHist, noZeroPosHist, maxGrayLevel, minGrayLevel, method = 'reverse'):

        hist = [0]*(maxGrayLevel+1)

        if method == 'reverse':
            oldIdx = self.genes[-1].position
            for i in range(len(noZeroPosHist)-1, -1, -1):
                idx = self.genes[i].position
                if idx < minGrayLevel or idx > maxGrayLevel:
                    print 'idx', idx
                    exit()
                ind = noZeroPosHist[i]
                if idx == oldIdx:
                    hist[idx] += targetHist[ind]
                else:
                    hist[idx] = targetHist[ind]
                    oldIdx = self.genes[i].position
        
        elif method == 'direct':
            oldIdx = self.genes[0].position
            for i in range(0, len(noZeroPosHist)):
                idx = self.genes[i].position
                if idx < minGrayLevel or idx > maxGrayLevel:
                    print 'idx', idx
                    exit()
                ind = noZeroPosHist[i]
                if idx == oldIdx:
                    hist[idx] += targetHist[ind]
                else:
                    hist[idx] = targetHist[ind]
                    oldIdx = self.genes[i].position

        opt_T, mu1, mu2 = self.__optimalThreshold(hist, 0.001, 100)

        sigma1, sigma2, halfWidth1, halfWidth2= self.__calculateVariances(hist, opt_T, mu1, mu2)

        self.__term1 = abs(2*opt_T - mu1 - mu2)
        self.__term2 = abs( halfWidth1*0.33 - sigma1 )
        self.__term3 = abs( halfWidth2*0.33 - sigma2 )

        dist =  self.__term1 +  self.__term2 +  self.__term3

        return dist, opt_T, hist

    def getFitness(self):
        return self.__fitness

    def getOpt_T(self):
        return self.__opt_T

    def getMatrix(self):
        return self.__matrix

# Class representing a gene of each individual      
class gene(object):
    
    def __init__(self, pos = 0):

        # Each gene is a bin of the histogram
        self.position = pos
    
    # Mutate the bin index
    def mutatePosition(self, minGrayLevel, maxGrayLevel, opt_T):

        if self.position <= opt_T:
            value = rnd.randint(minGrayLevel, opt_T)
        else:
            value = rnd.randint(opt_T, maxGrayLevel)
        
        if value > maxGrayLevel:
            value = maxGrayLevel
        elif value < minGrayLevel:
            value = minGrayLevel
        
        self.position = value

# Class containing the genetic operators (crossover and mutation)            
class geneticOperation(object):

    # Mutation of the genes
    def mutate(self, genes, minGrayLevel, maxGrayLevel, opt_T, rate):
        
        for i in range(0, len(genes)):
            if rnd.uniform(0, 1) < rate:
                genes[i].mutatePosition(minGrayLevel, maxGrayLevel, opt_T)
                
    def crossoverSingle(self, parent_1, parent_2):

        numberGenes = len(parent_1.genes)
        randNum = rnd.randint(0, numberGenes)
        
        list1 = deepcopy(parent_1.genes[0:randNum])
        list2 = deepcopy(parent_2.genes[randNum:numberGenes])

        return list1 + list2

    # Uniform and circular crossover
    def crossoverUniform(self, parent_1, parent_2, cross_point):

        # If the crossover point exists, it is used to mix the genes of the two parents
        if cross_point:
            numberGenes = len(parent_1.genes)
            list1 = []
            half = int(round( numberGenes / 2.0))
            if cross_point >= half:
                list1 = deepcopy(parent_2.genes[0:cross_point-half]) + deepcopy(parent_1.genes[cross_point-half:cross_point]) + deepcopy(parent_2.genes[cross_point:numberGenes])
            else:
                list1 = deepcopy(parent_1.genes[0:cross_point]) + deepcopy(parent_2.genes[cross_point:cross_point+half]) + deepcopy(parent_1.genes[cross_point+half:numberGenes])
            return list1, cross_point
        
        # If the crossover point does not exist, it is randomly selected to mix the genes of the two parents
        else:
            numberGenes = len(parent_1.genes)
            randNum = rnd.randint(0, numberGenes-1)
            list1 = []
            half = int(round(numberGenes / 2.0))

            if randNum >= half:
                list1 = deepcopy(parent_1.genes[0:randNum-half]) + deepcopy(parent_2.genes[randNum-half:randNum]) + deepcopy(parent_1.genes[randNum:numberGenes]) 
            else:
                list1 = deepcopy(parent_2.genes[0:randNum]) + deepcopy(parent_1.genes[randNum:randNum+half]) + deepcopy(parent_2.genes[randNum+half:numberGenes])
            return list1, randNum