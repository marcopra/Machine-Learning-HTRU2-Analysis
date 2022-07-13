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
    prior_probability = 0.5
    k = 5 #k is the number of splitting for the k-fold
    prob_cost = [[0.5,1,1], [0.9, 1, 1], [0.1, 1, 1]] #Prbability and Costs
    gaussianization_test = [False, True]
    gaussianization_test = [False]



    for k in [5]:#, None]:
        print(f'----------------------K-FOLD = {k} -----------------------')

        for gaussianization in gaussianization_test:
            if not gaussianization:
                print(f'__________________________RAW FEATURES__________________________')

            else :
                print(f'__________________________GAUSSIANIZATION__________________________')

            for m in [None, 7, 6]:
                if m is None:
                    print(f'----------------------NO PCA-----------------------')
                else:
                    print(f'----------------------PCA = {m}-----------------------')

                ''' MVG Classifier'''

                print("-------------MVG Full Cov------------")
            
                print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))

                for p, cfn, cfp in prob_cost:
                    if k is None:
                        (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

                        
                        if gaussianization:
                            DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                        if m is not None:
                            P = preprocessing.compute_pca(DTR, m)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)

                        acc, llr = classifier.mvg(DTR, LTR, DTE, LTE)

                        minDCF = compute_min_DCF(llr, LTE, p, cfn, cfp)
                        actDCF = compute_act_DCF(llr, LTE, p, cfn, cfp)

                    else:
                        scores = numpy.array([])
                        LTE_kfold = numpy.array([])
                        for i in range(0,k):
                            DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                            

                            if gaussianization:
                                DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                            if m is not None:
                                P = preprocessing.compute_pca(DTR, m)
                                DTR = numpy.dot(P.T, DTR)
                                DTE = numpy.dot(P.T, DTE)

                            acc,llr =  classifier.mvg(DTR,LTR,DTE,LTE)
                            scores = numpy.concatenate((scores, llr))
                            LTE_kfold = numpy.concatenate((LTE_kfold, LTE))


                        minDCF = compute_min_DCF(scores,LTE_kfold, p, cfn, cfp)
                        actDCF = compute_act_DCF(scores,LTE_kfold, p, cfn, cfp)

                    print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
                print('\n\n\n\n')

                ''' MVG Classifier Diag'''
                
                print("-------------MVG Diag Cov------------")
               
                print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))
                
                for p, cfn, cfp in prob_cost:
                    if k is None:
                        (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
                        
                        if gaussianization:
                            DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                        if m is not None:
                            P = preprocessing.compute_pca(DTR, m)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)
                
                        acc, llr = classifier.mvg(DTR,LTR,DTE,LTE, diag = True)
                
                        minDCF = compute_min_DCF(llr, LTE, p, cfn, cfp)
                        actDCF = compute_act_DCF(llr, LTE, p, cfn, cfp)
                
                    else:
                        scores = numpy.array([])
                        LTE_kfold = numpy.array([])
                        for i in range(0,k):
                            DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                           
                            if gaussianization:
                                DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                            if m != 8:
                                P = preprocessing.compute_pca(DTR, m)
                                DTR = numpy.dot(P.T, DTR)
                                DTE = numpy.dot(P.T, DTE)
                            acc,llr =  classifier.mvg(DTR,LTR,DTE,LTE, diag = True)
                            scores = numpy.concatenate((scores, llr))
                            LTE_kfold = numpy.concatenate((LTE_kfold, LTE))
                
                        minDCF = compute_min_DCF(scores,LTE_kfold, p, cfn, cfp)
                        actDCF = compute_act_DCF(scores,LTE_kfold, p, cfn, cfp)
                
                    print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
                print('\n\n\n\n')


                ''' Tied MVG Classifier'''
                print("---------Tied MVG---------")
                
                print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))

                for p, cfn, cfp in prob_cost:
                    if k is None:
                        (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
                        
                        if gaussianization:
                            DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                        if m is not None:
                            P = preprocessing.compute_pca(DTR, m)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)

                        acc, llr = classifier.tied_mvg(DTR, LTR, DTE, LTE)

                        minDCF = compute_min_DCF(llr, LTE, p, cfn, cfp)
                        actDCF = compute_act_DCF(llr, LTE, p, cfn, cfp)

                    else:
                        scores = numpy.array([])
                        LTE_kfold = numpy.array([])
                        for i in range(0,k):
                            DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                            
                            if gaussianization:
                                DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                            if m != 8:
                                P = preprocessing.compute_pca(DTR, m)
                                DTR = numpy.dot(P.T, DTR)
                                DTE = numpy.dot(P.T, DTE)
                            acc,llr =  classifier.tied_mvg(DTR,LTR,DTE,LTE)
                            scores = numpy.concatenate((scores, llr))
                            LTE_kfold = numpy.concatenate((LTE_kfold, LTE))

                        minDCF = compute_min_DCF(scores,LTE_kfold, p, cfn, cfp)
                        actDCF = compute_act_DCF(scores,LTE_kfold, p, cfn, cfp)

                    print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
                print('\n\n\n\n')

                ''' Tied Diag MVG Classifier'''
                print("---------Tied Diag MVG---------")
    
                print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))

                for p, cfn, cfp in prob_cost:
                    if k is None:
                        (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
                        
                        if gaussianization:
                            DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                        if m is not None:
                            P = preprocessing.compute_pca(DTR, m)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)

                        acc, llr = classifier.tied_mvg(DTR, LTR, DTE, LTE, diag=True)

                        minDCF = compute_min_DCF(llr, LTE, p, cfn, cfp)
                        actDCF = compute_act_DCF(llr, LTE, p, cfn, cfp)

                    else:
                        scores = numpy.array([])
                        LTE_kfold = numpy.array([])
                        for i in range(0,k):
                            DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                           
                            if gaussianization:
                                DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                            if m != 8:
                                P = preprocessing.compute_pca(DTR, m)
                                DTR = numpy.dot(P.T, DTR)
                                DTE = numpy.dot(P.T, DTE)
                            acc,llr =  classifier.tied_mvg(DTR,LTR,DTE,LTE, diag = True)
                            scores = numpy.concatenate((scores, llr))
                            LTE_kfold = numpy.concatenate((LTE_kfold, LTE))

                        minDCF = compute_min_DCF(scores,LTE_kfold, p, cfn, cfp)
                        actDCF = compute_act_DCF(scores,LTE_kfold, p, cfn, cfp)

                    print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
                print('\n\n\n\n')

                ''' Naive Bayes Classifier'''
                print("---------Naive Bayes---------")

                print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))

                for p, cfn, cfp in prob_cost:
                    if k is None:
                        (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
                        
                        if gaussianization:
                            DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                        if m is not None:
                            P = preprocessing.compute_pca(DTR, m)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)

                        acc, llr = classifier.naive_bayes(DTR, LTR, DTE, LTE)

                        minDCF = compute_min_DCF(llr, LTE, p, cfn, cfp)
                        actDCF = compute_act_DCF(llr, LTE, p, cfn, cfp)

                    else:
                        scores = numpy.array([])
                        LTE_kfold = numpy.array([])
                        for i in range(0,k):
                            DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                            
                            if gaussianization:
                                DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                            if m != 8:
                                P = preprocessing.compute_pca(DTR, m)
                                DTR = numpy.dot(P.T, DTR)
                                DTE = numpy.dot(P.T, DTE)
                            acc,llr =  classifier.naive_bayes(DTR,LTR,DTE,LTE)
                            scores = numpy.concatenate((scores, llr))
                            LTE_kfold = numpy.concatenate((LTE_kfold, LTE))

                        minDCF = compute_min_DCF(scores,LTE_kfold, p, cfn, cfp)
                        actDCF = compute_act_DCF(scores,LTE_kfold, p, cfn, cfp)

                    print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
                print('\n\n\n\n')

                ''' Tied Naive Bayes Classifier'''
                print("---------Tied Naive Bayes---------")

                
                print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format('minDCF','actDCF', 'p','cfp', 'cfn'))

                for p, cfn, cfp in prob_cost:
                    if k is None:
                        (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
                        
                        if gaussianization:
                            DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                        if m is not None:
                            P = preprocessing.compute_pca(DTR, m)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)

                        acc, llr = classifier.tied_naive_bayes(DTR, LTR, DTE, LTE)

                        minDCF = compute_min_DCF(llr, LTE, p, cfn, cfp)
                        actDCF = compute_act_DCF(llr, LTE, p, cfn, cfp)

                    else:
                        scores = numpy.array([])
                        LTE_kfold = numpy.array([])
                        for i in range(0,k):
                            DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                            
                            if gaussianization:
                                DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                            if m != 8:
                                P = preprocessing.compute_pca(DTR, m)
                                DTR = numpy.dot(P.T, DTR)
                                DTE = numpy.dot(P.T, DTE)
                            acc,llr =  classifier.naive_bayes(DTR,LTR,DTE,LTE)
                            scores = numpy.concatenate((scores, llr))
                            LTE_kfold = numpy.concatenate((LTE_kfold, LTE))

                        minDCF = compute_min_DCF(scores,LTE_kfold, p, cfn, cfp)
                        actDCF = compute_act_DCF(scores,LTE_kfold, p, cfn, cfp)

                    print ("{:<25} {:<25} {:<8} {:<8} {:<8}".format(minDCF, actDCF, p, cfp, cfn))
                print('\n\n\n\n')


                if k is None:
                    continue

                ''' Linear Logistic Regression'''
                print("---------Logistic Regression---------")

                #We use here the k-fold cross validation
            
                lamb = [1e-4]

                pi_t = [0.5 , 0.9, 0.1] #pi for calibration

                print ("{:<25} {:<25} {:<25} {:<8}".format('', 'minDCF','actDCF', 'p' ))

                for p, cfn, cfp in prob_cost:
                    for l in lamb:
                        for p_cal in pi_t:
                            scores = numpy.array([])
                            LTE_kfold = numpy.array([])
                            for i in range(0,k):
                                DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                                

                                if gaussianization:
                                    DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                                if m != 8:
                                    P = preprocessing.compute_pca(DTR, m)
                                    DTR = numpy.dot(P.T, DTR)
                                    DTE = numpy.dot(P.T, DTE)

                                acc,llr =  classifier.compute_logreg(DTR,LTR,DTE,LTE, l, p_cal)

                                scores = numpy.concatenate((scores, llr))
                                LTE_kfold = numpy.concatenate((LTE_kfold, LTE))


                            minDCF = compute_min_DCF(scores, L, p, cfn, cfp)
                            actDCF = compute_act_DCF(scores, L, p, cfn, cfp)


                            print ("{:<25} {:<25} {:<25} {:<8}".format(f'LogReg(λ={l}, πt={p_cal})',minDCF, actDCF, p))

            

                print('\n\n\n\n')


                ''' Quadratic Logistic Regression'''
                print("---------Quadratic Logistic Regression---------")

                #We use here the k-fold cross validation
            

                lamb = [1e-5]
                pi_t = [0.5 , 0.9, 0.1] #pi for calibration

               
                print ("{:<25} {:<25} {:<25} {:<8}".format('', 'minDCF','actDCF', 'p' ))

                for p, cfn, cfp in prob_cost:
                    for l in lamb:
                        for p_cal in pi_t:
                            scores = numpy.array([])
                            LTE_kfold = numpy.array([])
                            for i in range(0,k):
                                DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                                
                                if gaussianization:
                                    DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                                if m != 8:
                                    P = preprocessing.compute_pca(DTR, m)
                                    DTR = numpy.dot(P.T, DTR)
                                    DTE = numpy.dot(P.T, DTE)
                                DTR = preprocessing.feature_expansion(DTR)
                                DTE = preprocessing.feature_expansion(DTE)
                                acc,llr =  classifier.compute_logreg(DTR,LTR,DTE,LTE, l, p_cal)
                                scores = numpy.concatenate((scores, llr))
                                LTE_kfold = numpy.concatenate((LTE_kfold, LTE))


                            minDCF = compute_min_DCF(scores,L, p, cfn, cfp)
                            actDCF = compute_act_DCF(scores,L, p, cfn, cfp)



                            print ("{:<25} {:<25} {:<25} {:<8}".format(f'QuadLogReg(λ={l}, πt={p_cal})',minDCF, actDCF, p))

                print('\n\n\n\n')


                '''Linear SVM'''
                print("---------Linear SVM---------")

                prob_cost = [[0.5, 1, 1], [0.9, 1, 1], [0.1, 1, 1]]
                C = [0.1]
                pi_cal = [None, 0.5, 0.9, 0.1]

                print("{:<25} {:<25} {:<25} {:<8}".format('', 'minDCF', 'actDCF', 'p'))
                for pit in pi_cal:

                    for p, cfn, cfp in prob_cost:
                        for c_ in C:

                            scores = numpy.array([])
                            for i in range(0, k):
                                DTR, LTR, DTE, LTE = kfold_validation(D, L, i, k)
                                
                                if gaussianization:
                                    DTR, DTE = preprocessing.gaussianization(DTR, DTE)

                                if m is not None:
                                    P = preprocessing.compute_pca(DTR, m)
                                    DTR = numpy.dot(P.T, DTR)
                                    DTE = numpy.dot(P.T, DTE)

                                acc, llr = classifier.train_SVM_linear(DTR, LTR, DTE, LTE, c_, pi=pit)

                                scores = numpy.concatenate((scores, llr))

                            minDCF = compute_min_DCF(scores, L, p, cfn, cfp)
                            actDCF = compute_act_DCF(scores, L, p, cfn, cfp)
                            print("{:<25} {:<25} {:<25} {:<8}".format(f'SVM(C={c_}, πt={pit})', minDCF, actDCF, p))


                print('\n\n\n\n')

                '''Polynomial SVM'''
                print("---------Polynomial SVM---------")

                prob_cost = [[0.5, 1, 1], [0.9, 1, 1], [0.1, 1, 1]]
                C = 0.01
                pi_cal = [None, 0.5, 0.9, 0.1]
                c_small = 10


                print("{:<45} {:<25} {:<25} {:<8}".format('', 'minDCF', 'actDCF', 'p'))
                for pit in pi_cal:

                    for p, cfn, cfp in prob_cost:
                        

                        scores = numpy.array([])
                        for i in range(0, k):
                            DTR, LTR, DTE, LTE = kfold_validation(D, L, i, k)
                            
                            if gaussianization:
                                DTR, DTE = preprocessing.gaussianization(DTR, DTE)

                            if m is not None:
                                P = preprocessing.compute_pca(DTR, m)
                                DTR = numpy.dot(P.T, DTR)
                                DTE = numpy.dot(P.T, DTE)

                            acc,llr =  classifier.train_SVM_kernel(DTR, LTR, DTE, LTE, C = C, K = 1, d = 2, c = c_small, pi = pit)

                            scores = numpy.concatenate((scores, llr))

                        minDCF = compute_min_DCF(scores, L, p, cfn, cfp)
                        actDCF = compute_act_DCF(scores, L, p, cfn, cfp)
                        print ("{:<45} {:<25} {:<25} {:<8}".format(f'Poly SVM(C={C}, d = 2, K = 1, c = {c_small}, πt={pit})',minDCF, actDCF, p))


                print('\n\n\n\n')
        

                
                '''RBF SVM'''

                print("---------RBF SVM---------")

        
                C = 1e-1
                g = 1e-3
                pi_t = [None, 0.5 , 0.9, 0.1] #pi for calibration

                
                print ("{:<25} {:<25} {:<25} {:<8}".format('', 'minDCF','actDCF', 'p' ))

                for p, cfn, cfp in prob_cost:
            
                    for p_cal in pi_t:
                        scores = numpy.array([])
                        for i in range(0,k):
                            DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                           

                            if gaussianization:
                                DTR, DTE = preprocessing.gaussianization(DTR, DTE)
                            if m is not None:
                                P = preprocessing.compute_pca(DTR, m)
                                DTR = numpy.dot(P.T, DTR)
                                DTE = numpy.dot(P.T, DTE)

                            acc,llr =  classifier.train_SVM_kernel(DTR, LTR, DTE, LTE, gamma = g , C = C, RBF = True, pi = p_cal)

                            scores = numpy.concatenate((scores, llr))

                        minDCF = compute_min_DCF(scores, L, p, cfn, cfp)
                        actDCF = compute_act_DCF(scores, L, p, cfn, cfp)
                        print ("{:<25} {:<25} {:<25} {:<8}".format(f'RBF SVM(C={C},gamma = {g}, πt={p_cal})',minDCF, actDCF, p))

    
                print('\n\n\n\n')



                '''GMM '''
                print("---------GMM---------")

    
                type_covariances = {'Full' : 8, 'Tied' : 4, 'Diag' : 8, 'Tied-Diag':4}
                prob_cost = [[0.5, 1, 1], [0.9, 1, 1], [0.1, 1, 1]]
               
                
                print ("{:<25} {:<25} {:<25} {:<8}".format('', 'minDCF','actDCF', 'p' ))
                
                for p, cfn, cfp in prob_cost:
                    for type_covariance, components in type_covariances.items():
                        scores = numpy.array([])
                        for i in range(0, k):

                            DTR, LTR, DTE, LTE = kfold_validation(D, L, i, k)
                            

                            if gaussianization:
                                DTR, DTE = preprocessing.gaussianization(DTR, DTE)

                            if m is not None:
                                P = preprocessing.compute_pca(DTR, m)
                                DTR = numpy.dot(P.T, DTR)
                                DTE = numpy.dot(P.T, DTE)


                            llr = classifier.computeGMM(DTR, LTR, DTE, LTE, 0.01 , components, 0.01, type_covariance)
    

                            scores = numpy.concatenate((scores, llr))


                        minDCF = compute_min_DCF(scores, L, p, cfn, cfp)
                        actDCF = compute_act_DCF(scores, L, p, cfn, cfp)


                        print("{:<25} {:<25} {:<25} {:<8} ".format(f'GMM(n = {components}, type = {type_covariance})', minDCF, actDCF, p))

                
                
                print('\n\n\n\n')

                

