import numpy
import scipy
import scipy.linalg
import pylab
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
from Evaluation import *
from Preprocessing import Preprocessing
from Classifier import Classifier
from utils import * 
from sklearn.utils import shuffle


        
if __name__ == '__main__':

    preprocessing = Preprocessing()
    classifier = Classifier()
    D,L = load("Train.txt")
    D,L = my_shuffle(D,L)
    # plot_features(D,L)
    D = preprocessing.z_normalization(D)
    # plot_features(D,L, "z_normalized_")  
    prior_probability = 0.5
    k = 5 #k is the number of spitting for the k-fold

    # D = preprocessing.gaussianization(D)
    # plot_features(D,L, "gaussianized_")  
    
    
    
    #----------------GRAPH---------------------------

    plot_minDCF_logReg(classifier, preprocessing, D,L)

    plot_minDCF_logReg(classifier, preprocessing, D,L, quadratic = True)

    plot_minDCF_linearSVM(classifier, preprocessing, D,L)
    
    plot_minDCF_RBF(classifier, preprocessing, D,L)

    plot_minDCF_poly(classifier, preprocessing, D,L)

    plot_minDCF_GMM(classifier, preprocessing, D,L)

    plotGMM_bar(classifier, preprocessing, D,L)

    
    