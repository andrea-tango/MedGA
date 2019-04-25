import numpy as np
import scipy.misc
import math
import os

# Class containing image processing functions
class processing(object):
    
    # Loading the input image
    def loadImage(self, target_img_name, pathOut):
    
        image = scipy.misc.imread(target_img_name)

        maxValue = np.max(image)

        hist, _ = np.histogram(image, bins = maxValue+1)

        hist[0] = 0

        posNoZeros = list(np.nonzero(hist)[0])

        scipy.misc.imsave(pathOut + os.sep + 'imageOriginal.png', image)

        T_k =  self.__optimalThreshold(hist, 0.001, 100)
            
        return image, len(posNoZeros), hist, posNoZeros, maxValue, T_k

    # Optimal Threshold algorithm
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
            
        return T_k