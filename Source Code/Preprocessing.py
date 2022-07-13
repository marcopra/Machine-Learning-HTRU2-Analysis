import numpy
import scipy
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.linalg import norm
from scipy.stats import norm, rankdata
from utils import * 

class Preprocessing():
    
    def __init__(self):
        pass

    def z_normalization(self, D):

        for i in range(D.shape[0]):
            D[i, : ]  = (D[i,: ] - numpy.mean(D[i, :]))/(numpy.std(D[i, : ]))

        return D
    
    def z_normalization_split(self, DTR, DTE):
        
        DTR_mean = compute_empirical_mean(DTR)
        DTR_std = mcol(numpy.std(DTR, axis = 1))
        DTR = (DTR - DTR_mean)/DTR_std
        DTE = (DTE - DTR_mean)/DTR_std
            
        return DTR, DTE


    def compute_lda(self, SB, SW, m = 1):

        s, U = scipy.linalg.eig(SB, SW)
        W = U[:, ::-1][:, 0:m]

        return W

    def compute_pca(self, D, m):
        #data centered
        DC = D - mcol(D.mean(1))

        #C -> covariance matrix
        C = numpy.dot(DC, DC.T)
        C = C/ D.shape[1]

        s, U = numpy.linalg.eigh(C)
    
        P = U[: , ::-1][:, 0 : m]
        
        return P
    
    def gaussianization(self, DTR, DTE = None):

        rankDTR = numpy.zeros(DTR.shape)
        for i in range(DTR.shape[0]):
            for j in range(DTR.shape[1]):
                rankDTR[i][j] = (DTR[i] < DTR[i][j]).sum()
        
        DTR_new = scipy.stats.norm.ppf((rankDTR + 1)/(DTR.shape[1] + 2))

        if DTE is not None :
            rankDTE = numpy.zeros(DTE.shape)
            for i in range(DTE.shape[0]):
                for j in range(DTE.shape[1]):
                    rankDTE[i][j] = (DTR[i] < DTE[i][j]).sum()

            DTE_new = scipy.stats.norm.ppf((rankDTE + 1)/(DTR.shape[1] + 2)) 

            return DTR_new, DTE_new
            
        return DTR_new     

    def feature_expansion(self, D):

        expanded_features = []

        for i in range(D.shape[1]):
            vec = numpy.reshape(numpy.dot(mcol(D[:, i]), mrow(D[:, i])), (D.shape[0]**2,1), order = 'F')
            expanded_features.append(vec)

        D = numpy.vstack((numpy.hstack(expanded_features), D))
        return D

