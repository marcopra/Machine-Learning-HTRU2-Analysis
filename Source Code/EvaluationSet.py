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
    DTR,LTR = load("Train.txt")
    DTE,LTE = load("Test.txt")
    prob_cost = [[0.5,1,1], [0.1, 1, 1], [0.9, 1, 1]] #Prbability and Costs
    
    DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)

    for m in [None, 7]:
        if m is None:
            print(f'----------------------NO PCA-----------------------')
        else:
            print(f'----------------------PCA = {m}-----------------------')

        ''' MVG Classifier'''
        print("-------------MVG Full Cov------------")
        print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
        for p, cfn, cfp in prob_cost:
            DTR, LTR = load("Train.txt")
            DTE, LTE = load("Test.txt")
            DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)
                     
            if m is not None:
                P = preprocessing.compute_pca(DTR, m)
                DTR = numpy.dot(P.T, DTR)
                DTE = numpy.dot(P.T, DTE)

            _, scores =  classifier.mvg(DTR,LTR,DTE,LTE)
            

            minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
            actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)

            print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
        print('\n\n\n\n')

        ''' MVG Classifier Diag'''
        print("-------------MVG Diag Cov------------")
        print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
        for p, cfn, cfp in prob_cost:
            DTR, LTR = load("Train.txt")
            DTE, LTE = load("Test.txt")
            DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)

            if m is not None:
                P = preprocessing.compute_pca(DTR, m)
                DTR = numpy.dot(P.T, DTR)
                DTE = numpy.dot(P.T, DTE)

            _, scores = classifier.mvg(DTR,LTR,DTE,LTE, diag = True)
            

            minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
            actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)

            print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
        print('\n\n\n\n')


        ''' Tied MVG Classifier'''
        print("---------Tied MVG---------")
        print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
        for p, cfn, cfp in prob_cost:
            DTR, LTR = load("Train.txt")
            DTE, LTE = load("Test.txt")
            DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)

            if m is not None:
                P = preprocessing.compute_pca(DTR, m)
                DTR = numpy.dot(P.T, DTR)
                DTE = numpy.dot(P.T, DTE)

            _, scores = classifier.tied_mvg(DTR,LTR,DTE,LTE)
            

            minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
            actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)

            print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
        print('\n\n\n\n')

        ''' Tied Diag MVG Classifier'''
        print("---------Tied Diag MVG---------")
        print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
        for p, cfn, cfp in prob_cost:
            DTR, LTR = load("Train.txt")
            DTE, LTE = load("Test.txt")
            DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)

            if m is not None:
                P = preprocessing.compute_pca(DTR, m)
                DTR = numpy.dot(P.T, DTR)
                DTE = numpy.dot(P.T, DTE)

            _, scores  =  classifier.tied_mvg(DTR, LTR, DTE, LTE, diag=True)
            

            minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
            actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)

            print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
        print('\n\n\n\n')

        ''' Naive Bayes Classifier'''
        print("---------Naive Bayes---------")
        print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
        for p, cfn, cfp in prob_cost:
            DTR, LTR = load("Train.txt")
            DTE, LTE = load("Test.txt")
            DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)
            
            if m is not None:
                P = preprocessing.compute_pca(DTR, m)
                DTR = numpy.dot(P.T, DTR)
                DTE = numpy.dot(P.T, DTE)

            _, scores =  classifier.naive_bayes(DTR, LTR, DTE, LTE)
            

            minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
            actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)

            print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
        print('\n\n\n\n')

        ''' Tied Naive Bayes Classifier'''
        print("---------Tied Naive Bayes---------")
        print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
        for p, cfn, cfp in prob_cost:
            DTR, LTR = load("Train.txt")
            DTE, LTE = load("Test.txt")
            DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)

            if m is not None:
                P = preprocessing.compute_pca(DTR, m)
                DTR = numpy.dot(P.T, DTR)
                DTE = numpy.dot(P.T, DTE)

            _, scores =  classifier.tied_naive_bayes(DTR, LTR, DTE, LTE)
            

            minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
            actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)

            print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
        print('\n\n\n\n')

        ''' Linear Logistic Regression'''
        print("---------Logistic Regression---------")
        print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
        for p, cfn, cfp in prob_cost:
            DTR, LTR = load("Train.txt")
            DTE, LTE = load("Test.txt")
            DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)

            if m is not None:
                P = preprocessing.compute_pca(DTR, m)
                DTR = numpy.dot(P.T, DTR)
                DTE = numpy.dot(P.T, DTE)

            _, scores = classifier.compute_logreg(DTR,LTR,DTE,LTE, 1e-4, 0.5)
            

            minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
            actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)

            print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
        print('\n\n\n\n')

        ''' Quadratic Logistic Regression'''
        print("---------Quadratic Logistic Regression---------")
        print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
        for p, cfn, cfp in prob_cost:
            DTR, LTR = load("Train.txt")
            DTE, LTE = load("Test.txt")
            DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)

            if m is not None:
                P = preprocessing.compute_pca(DTR, m)
                DTR = numpy.dot(P.T, DTR)
                DTE = numpy.dot(P.T, DTE)

            DTR = preprocessing.feature_expansion(DTR)
            DTE = preprocessing.feature_expansion(DTE)
            _, scores =  classifier.compute_logreg(DTR,LTR,DTE,LTE, 1e-4, 0.5)
            

            minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
            actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)

            print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
        print('\n\n\n\n')


        '''Linear SVM'''
        print("---------Linear SVM---------")
        print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
        for p, cfn, cfp in prob_cost:
            DTR, LTR = load("Train.txt")
            DTE, LTE = load("Test.txt")
            DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)

            if m is not None:
                P = preprocessing.compute_pca(DTR, m)
                DTR = numpy.dot(P.T, DTR)
                DTE = numpy.dot(P.T, DTE)

            _, scores = classifier.train_SVM_linear(DTR, LTR, DTE, LTE, 0.5, pi=None)
            

            minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
            actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)

            print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
        print('\n\n\n\n')

        '''Polynomial SVM'''
        print("---------Polynomial SVM---------")
        print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
        for p, cfn, cfp in prob_cost:
            DTR, LTR = load("Train.txt")
            DTE, LTE = load("Test.txt")
            DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)

            if m is not None:
                P = preprocessing.compute_pca(DTR, m)
                DTR = numpy.dot(P.T, DTR)
                DTE = numpy.dot(P.T, DTE)

            _, scores = classifier.train_SVM_kernel(DTR, LTR, DTE, LTE, C = 0.01, K = 1, d = 2, c = 1, pi = 0.5)
            

            minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
            actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)

            print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
        print('\n\n\n\n')


        '''RBF SVM'''
        print("---------RBF SVM---------")
        print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
        for p, cfn, cfp in prob_cost:
            DTR, LTR = load("Train.txt")
            DTE, LTE = load("Test.txt")
            DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)

            if m is not None:
                P = preprocessing.compute_pca(DTR, m)
                DTR = numpy.dot(P.T, DTR)
                DTE = numpy.dot(P.T, DTE)

            _, scores = classifier.train_SVM_kernel(DTR, LTR, DTE, LTE, gamma = 1e-1 , C = 0.1, RBF = True, pi = 0.5)
            

            minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
            actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)

            print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))

        print('\n\n\n\n')

        '''GMM '''
        print("---------GMM---------")
        

        components = 8
        type_covariances = ['Full', 'Tied', 'Diag']
        prob_cost = [[0.5, 1, 1], [0.9, 1, 1], [0.1, 1, 1]]
        
        
        print ("{:<25} {:<25} {:<25} {:<8}".format('', 'minDCF','actDCF', 'p' ))
        
        for p, cfn, cfp in prob_cost:
            for type_covariance in type_covariances:
                DTR, LTR = load("Train.txt")
                DTE, LTE = load("Test.txt")
                DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)
            
                if m is not None:
                    P = preprocessing.compute_pca(DTR, m)
                    DTR = numpy.dot(P.T, DTR)
                    DTE = numpy.dot(P.T, DTE)


                scores = classifier.computeGMM(DTR, LTR, DTE, LTE, 0.01 , components, 0.01, type_covariance)

                minDCF = compute_min_DCF(scores,LTE, p, cfn, cfp)
                actDCF = compute_act_DCF(scores,LTE, p, cfn, cfp)


                print("{:<25} {:<25} {:<25} {:<8} ".format(f'GMM(n = {components}, type = {type_covariance})', minDCF, actDCF, p))
        print('\n\n\n\n')