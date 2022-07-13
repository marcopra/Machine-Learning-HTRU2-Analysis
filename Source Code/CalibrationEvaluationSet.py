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

    cfn = 1
    cfp = 1
    k = 5
    l = 1e-4
    m = 7
    probs = [0.5, 0.1, 0.9]


    #MVG tied
    ''' Tied MVG Classifier'''
    print("---------Tied MVG---------")

    print ("{:<25} {:<25} {:<25} {:<8}".format("",'minDCF','actDCF', 'p'))

    m = 7
        
    for p in probs:
    
        DTR, LTR = load("Train.txt")
        DTE, LTE = load("Test.txt")
                     

        P = preprocessing.compute_pca(DTR, 7)
        DTR = numpy.dot(P.T, DTR)
        DTE = numpy.dot(P.T, DTE)

        acc,scores =  classifier.tied_mvg(DTR,LTR,DTE,LTE)
        

        print ("{:<25} {:<25} {:<25} {:<8}".format("uncalibrated", compute_min_DCF(scores, LTE, p, 1, 1), compute_act_DCF(scores, LTE, p, 1, 1),p))

        if p == 0.5:
            error_plot(scores, LTE, "Tied-Cov", pi = p)

        _, scoreTR = classifier.tied_mvg(DTR,LTR,DTR,LTR)

        score_calibrated =  classifier.compute_logreg_calibration(mrow(scoreTR), LTR, mrow(scores), l, pi = p)

        if p == 0.5:
            scores1 = score_calibrated
            L1 = LTE
            error_plot(score_calibrated, LTE, "Tied-Cov", pi = p, calibrated = True)

        print ("{:<25} {:<25} {:<25} {:<8}".format("calibrated",compute_min_DCF(score_calibrated, LTE, p, 1, 1), compute_act_DCF(score_calibrated, LTE, p, 1, 1),p))



    ''' Linear Logistic Regression Classifier'''
    print("---------Linear Logistic Regression---------")
    for p in probs:
        scores = numpy.array([])
    
        DTR, LTR = load("Train.txt")
        DTE, LTE = load("Test.txt")
            
        DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)
            
        P = preprocessing.compute_pca(DTR, 7)
        DTR = numpy.dot(P.T, DTR)
        DTE = numpy.dot(P.T, DTE)

        acc,scores =  classifier.compute_logreg(DTR,LTR,DTE,LTE, 1e-4, 0.5)
    
            
        print ("{:<25} {:<25} {:<25} {:<8}".format("uncalibrated",compute_min_DCF(scores, LTE, p, 1, 1), compute_act_DCF(scores, LTE, p, 1, 1),p))
        
        if p == 0.5:
            error_plot(scores, LTE, "Linear Log Reg", pi = p)

        _, scoreTR = classifier.compute_logreg(DTR,LTR,DTR,LTR, 1e-4, 0.5)

        
        score_calibrated =  classifier.compute_logreg_calibration(mrow(scoreTR), LTR, mrow(scores), l, pi = 0.5)
        if p == 0.5:
            scores2 = score_calibrated
            L2 = LTE
            error_plot(score_calibrated, LTE, "Linear Log Reg", pi = p, calibrated = True)
    
        print ("{:<25} {:<25} {:<25} {:<8}".format("calibrated",compute_min_DCF(score_calibrated, LTE, p, 1, 1), compute_act_DCF(score_calibrated, LTE, p, 1, 1),p))    


    ''' Linear SVM Classifier'''
    print("---------Linear SVM---------")
    for p in probs:
        
        DTR, LTR = load("Train.txt")
        DTE, LTE = load("Test.txt")
        DTR, DTE = preprocessing.z_normalization_split(DTR, DTE)
        
               
        P = preprocessing.compute_pca(DTR, 7)
        DTR = numpy.dot(P.T, DTR)
        DTE = numpy.dot(P.T, DTE)

        acc,scores =  classifier.train_SVM_linear(DTR, LTR, DTE, LTE, 0.5)

            
        print ("{:<25} {:<25} {:<25} {:<8}".format("uncalibrated",compute_min_DCF(scores, LTE, p, 1, 1), compute_act_DCF(scores, LTE, p, 1, 1),p))

        if p == 0.5:
            error_plot(scores, LTE, "Linear SVM", pi = p)

        _, scoreTR = classifier.train_SVM_linear(DTR, LTR, DTR, LTR, 0.5)
    
        
        score_calibrated =  classifier.compute_logreg_calibration(mrow(scoreTR), LTR, mrow(scores), l, pi = 0.5)
        if p == 0.5:
            scores3 = score_calibrated
            L3 = LTE
            error_plot(score_calibrated, LTE, "Linear SVM", pi = p, calibrated = True)
    
        print ("{:<25} {:<25} {:<25} {:<8}".format("calibrated",compute_min_DCF(score_calibrated, LTE, p, 1, 1), compute_act_DCF(score_calibrated, LTE, p, 1, 1),p))    

 

    plotROC(scores1, scores2, scores3, L1, L2, L3)
    plotDET(scores1, scores2, scores3, L1, L2, L3)
