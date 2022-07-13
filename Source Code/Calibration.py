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
    D = preprocessing.z_normalization(D)

    cfn = 1
    cfp = 1
    k = 5
    l = 1e-5
    m = 7
    probs = [0.5, 0.1, 0.9]


    #MVG tied
    ''' Tied MVG Classifier'''
    print("---------Tied MVG---------")

    print ("{:<25} {:<25} {:<25} {:<8}".format("",'minDCF','actDCF', 'p'))


    for p in probs:
        scores = numpy.array([])

        for i in range(0,k):
            DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)


            P = preprocessing.compute_pca(DTR, 7)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)

            acc,llr =  classifier.tied_mvg(DTR,LTR,DTE,LTE)
            scores = numpy.concatenate((scores, llr))

        print ("{:<25} {:<25} {:<25} {:<8}".format("uncalibrated", compute_min_DCF(scores, L, p, 1, 1), compute_act_DCF(scores, L, p, 1, 1),p))

        error_plot(scores, L, "Tied-Cov", pi = p)

        scoreTR, LabelTR, scoreTE, LabelTE = score_shuffle_and_split(scores, L)

        score_calibrated =  classifier.compute_logreg_calibration(mrow(scoreTR), LabelTR, mrow(scoreTE), l, pi = p)

        error_plot(score_calibrated, LabelTE, "Tied-Cov", pi = p, calibrated = True)

        print ("{:<25} {:<25} {:<25} {:<8}".format("calibrated",compute_min_DCF(score_calibrated, LabelTE, p, 1, 1), compute_act_DCF(score_calibrated, LabelTE, p, 1, 1),p))



    ''' Linear Logistic Regression Classifier'''
    print("---------Linear Logistic Regression---------")
    for p in probs:
    
        scores = numpy.array([])
        for i in range(0,k):
            DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
            
            
            P = preprocessing.compute_pca(DTR, 7)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)


            acc,llr =  classifier.compute_logreg(DTR,LTR,DTE,LTE, 1e-4, 0.5)
            scores = numpy.concatenate((scores, llr))
            
        print ("{:<25} {:<25} {:<25} {:<8}".format("uncalibrated",compute_min_DCF(scores, L, p, 1, 1), compute_act_DCF(scores, L, p, 1, 1),p))

        error_plot(scores, L, "Linear Log Reg", pi = p)

        scoreTR, LabelTR, scoreTE, LabelTE = score_shuffle_and_split(scores, L)

        score_calibrated =  classifier.compute_logreg_calibration(mrow(scoreTR), LabelTR, mrow(scoreTE), l, pi = 0.5)

        error_plot(score_calibrated, LabelTE, "Linear Log Reg", pi = p, calibrated = True)

        print ("{:<25} {:<25} {:<25} {:<8}".format("calibrated",compute_min_DCF(score_calibrated, LabelTE, p, 1, 1), compute_act_DCF(score_calibrated, LabelTE, p, 1, 1),p))    


    ''' Linear SVM Classifier'''
    print("---------Linear SVM---------")
    for p in probs:
        scores = numpy.array([])

        for i in range(0,k):
            DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
            
            
            P = preprocessing.compute_pca(DTR, 7)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)

    

            acc,llr =  classifier.train_SVM_linear(DTR, LTR, DTE, LTE, 0.5)

            scores = numpy.concatenate((scores, llr))
            
        print ("{:<25} {:<25} {:<25} {:<8}".format("uncalibrated",compute_min_DCF(scores, L, p, 1, 1), compute_act_DCF(scores, L, p, 1, 1),p))

        error_plot(scores, L, "Linear SVM", pi = p)

        scoreTR, LabelTR, scoreTE, LabelTE = score_shuffle_and_split(scores, L)
        
        score_calibrated =  classifier.compute_logreg_calibration(mrow(scoreTR), LabelTR, mrow(scoreTE), l, pi = 0.5)

        error_plot(score_calibrated, LabelTE, "Linear SVM", pi = p, calibrated = True)
    
        print ("{:<25} {:<25} {:<25} {:<8}".format("calibrated",compute_min_DCF(score_calibrated, LabelTE, p, 1, 1), compute_act_DCF(score_calibrated, LabelTE, p, 1, 1),p))    


    # '''Fusion'''
    # print("---------Fusion---------")

    # l = 1e-4
    # C = 0.1

    # pi_t = [0.5 , 0.9, 0.1] #pi for calibration
    # probs = [0.5]

    # print ("{:<25} {:<25} {:<25} {:<8}".format('', 'minDCF','actDCF', 'p' ))

    # for p in probs:

       
    #     scores1 = numpy.array([])
    #     scores2 = numpy.array([])
        
    #     for i in range(0,k):
    #         DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
            
    #         if m != 8:
    #             P = preprocessing.compute_pca(DTR, m)
    #             DTR = numpy.dot(P.T, DTR)
    #             DTE = numpy.dot(P.T, DTE)

    #         acc,llr =  classifier.compute_logreg(DTR,LTR,DTE,LTE, 1e-4, 0.5)

    #         scores1 = numpy.concatenate((scores1, llr))

    #         acc, llr = classifier.train_SVM_linear(DTR, LTR, DTE, LTE, C, pi=None)
    #         scores2 = numpy.concatenate((scores2, llr))
        
    #     scores = scores1 + scores2
        
    #     error_plot(scores, L, "Fusion", pi = p)
        
        

    #     scoreTR, LabelTR, scoreTE, LabelTE = score_shuffle_and_split(scores, L)
        
    #     score_calibrated =  classifier.compute_logreg_calibration(mrow(scoreTR), LabelTR, mrow(scoreTE), l, pi = 0.5)

    #     error_plot(score_calibrated, LabelTE, "Fusion", pi = p, calibrated = True)


    #     print ("{:<25} {:<25} {:<25} {:<8}".format("Fusion calibrated",compute_min_DCF(score_calibrated, LabelTE, p, 1, 1), compute_act_DCF(score_calibrated, LabelTE, p, 1, 1),p))
       