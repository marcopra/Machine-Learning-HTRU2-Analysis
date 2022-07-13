import numpy
import scipy.optimize
from scipy.linalg import norm
from sklearn.utils import shuffle
from utils import *


class Classifier():

    def __init__(self):
        pass

    #Multivariate Gaussian Classifier
    def mvg(self, DTR, LTR, DTE, LTE, p = 0.5, diag = False): #p is referred to class 0
        mu0 = compute_empirical_mean(DTR[:, LTR == 0])
        mu1 = compute_empirical_mean(DTR[:, LTR == 1])
    
        cov0 = compute_empirical_cov(DTR[:, LTR == 0])
        cov1 = compute_empirical_cov(DTR[:, LTR == 1])

        if diag == True:
            cov0 = cov0*numpy.eye(cov0.shape[0])
            cov1 = cov1*numpy.eye(cov1.shape[0])

        
        ll0 = logpdf_GAU_ND(DTE, mu0, cov0)
        ll1 = logpdf_GAU_ND(DTE, mu1, cov1)
        
        ll = numpy.vstack((ll0, ll1))

        llr = ll1 - ll0
        
        SJoint = numpy.vstack((numpy.exp(ll0)*p, numpy.exp(ll1)*(1-p)))

        SMarginal = mrow(SJoint.sum(0))

        SPost = SJoint/SMarginal


        pred = SPost.argmax(0)
        accuracy = (LTE == pred).sum()/LTE.size

        return accuracy, llr

    #Tied Multivariate Gaussian Classifier
    def tied_mvg(self, DTR, LTR, DTE, LTE, p = 0.5, diag = False):
        mu0 = compute_empirical_mean(DTR[:, LTR == 0])
        mu1 = compute_empirical_mean(DTR[:, LTR == 1])

        #tied covariance
        SW = compute_sw(DTR, LTR)

        if diag == True:
            SW = SW*numpy.eye(SW.shape[0])
            
        
        ll0 = logpdf_GAU_ND(DTE, mu0, SW)
        ll1 = logpdf_GAU_ND(DTE, mu1, SW)

        llr = ll1 - ll0
        
        SJoint = numpy.vstack((numpy.exp(ll0)*p, numpy.exp(ll1)*(1-p)))

        SMarginal = mrow(SJoint.sum(0))

        SPost = SJoint/SMarginal

        pred = SPost.argmax(0)
        accuracy = (LTE == pred).sum()/LTE.size

        return accuracy, llr

    def naive_bayes(self, DTR, LTR, DTE, LTE, p = 0.5):

        mu0 = compute_empirical_mean(DTR[:, LTR == 0])
        mu1 = compute_empirical_mean(DTR[:, LTR == 1])
    
        cov0 = compute_empirical_cov(DTR[:, LTR == 0])
        cov1 = compute_empirical_cov(DTR[:, LTR == 1])

        #Diagonalize the covarinces matrix to obtain the Naive Bayes Classifier
        cov0 = cov0*numpy.identity(cov0.shape[0])
        cov1 = cov1*numpy.identity(cov1.shape[0])

        
        ll0 = logpdf_GAU_ND(DTE, mu0, cov0)
        ll1 = logpdf_GAU_ND(DTE, mu1, cov1)

        llr = ll1 - ll0
        

        ll = numpy.vstack((ll0, ll1))
        SJoint = numpy.vstack((numpy.exp(ll0)*p, numpy.exp(ll1)*(1-p)))

        SMarginal = mrow(SJoint.sum(0))

        SPost = SJoint/SMarginal

        pred = SPost.argmax(0)
        accuracy = (LTE == pred).sum()/LTE.size

       

        return accuracy, llr

    def tied_naive_bayes(self, DTR, LTR, DTE, LTE, p = 0.5):

        mu0 = compute_empirical_mean(DTR[:, LTR == 0])
        mu1 = compute_empirical_mean(DTR[:, LTR == 1])

        #tied covariance
        SW = compute_sw(DTR, LTR)


        #Diagonalize the covarinces matrix to obtain the Naive Bayes Classifier
        SW = SW*numpy.identity(SW.shape[0])
        

        ll0 = logpdf_GAU_ND(DTE, mu0, SW)
        ll1 = logpdf_GAU_ND(DTE, mu1, SW)

        llr = ll1 - ll0
        
    
        ll = numpy.vstack((ll0, ll1))

        SJoint = numpy.vstack((numpy.exp(ll0)*p, numpy.exp(ll1)*(1-p)))


        SMarginal = mrow(SJoint.sum(0))

        SPost = SJoint/SMarginal

        pred = SPost.argmax(0)
        accuracy = (LTE == pred).sum()/LTE.size


        return accuracy, llr

    #Logistic regression
    def logreg_obj_wrap(self, DTR, LTR, l, pi):
        M = DTR.shape[0]
        Z = LTR*2.0 - 1.0


        def logreg_obj(v):
            w, b = mcol(v[0:M]), v[-1]
            c1 = 0.5 * l * (numpy.linalg.norm(w) ** 2)
            c2 = ((pi) * (LTR[LTR == 1].shape[0]  ** -1)) * numpy.logaddexp(0, -Z[Z == 1] * (numpy.dot(w.T, DTR[:, LTR == 1]) + b)).sum()
            c3 = ((1 - pi) * (LTR[LTR == 0].shape[0] ** -1)) * numpy.logaddexp(0, -Z[Z == -1] * (numpy.dot(w.T, DTR[:, LTR == 0]) + b)).sum()
            return c1 + c2 + c3
    
        return logreg_obj

        # def logreg_obj( v):
        #     w = mcol(v[0:M])
        #     b = v[-1]

        #     S = numpy.dot(w.T, DTR) + b
        #     cxe = numpy.logaddexp(0, -S*Z)

        #     return numpy.linalg.norm(w)**2*l/2.0 + cxe.mean()
        # return logreg_obj

    def compute_logreg(self, DTR, LTR, DTE, LTE, lamb, pi = 0.5):
        logRegObj = self.logreg_obj_wrap(DTR, LTR, lamb, pi)


        _v,_J,_d = scipy.optimize.fmin_l_bfgs_b(logRegObj, numpy.zeros(DTR.shape[0] + 1), approx_grad = True)
        _w = _v[0: DTR.shape[0]]
        _b = _v[-1]

        
        STE = numpy.dot(_w.T, DTE) + _b
        LP = STE > 0
        acc = (LP ==LTE).sum() /LTE.size
        

        scores = numpy.dot(_w.T, DTE) + _b


        return acc, scores

    #Logistic regression
    def logreg_obj_wrap_calibration(self, DTR, LTR, l, pi):

        M = DTR.shape[0]
        Z = LTR * 2.0 - 1.0

        def logreg_obj(v):
            w, b = mcol(v[0:M]), v[-1]
            c1 = 0.5 * l * (numpy.linalg.norm(w) ** 2)
            c2 = ((pi) * (LTR[LTR == 1].shape[0] ** -1)) * numpy.logaddexp(0, -Z[Z == 1] * (
                        numpy.dot(w.T, DTR[:, LTR == 1]) + b)).sum()
            c3 = ((1 - pi) * (LTR[LTR == 0].shape[0] ** -1)) * numpy.logaddexp(0, -Z[Z == -1] * (
                        numpy.dot(w.T, DTR[:, LTR == 0]) + b)).sum()
            return c1 + c2 + c3

        return logreg_obj

    def compute_logreg_calibration(self, DTR, LTR, DTE, lamb, pi = 0.5):
        logRegObj = self.logreg_obj_wrap_calibration(DTR, LTR, lamb, pi)
        M = DTR.shape[0]
        K = LTR.max() +1

        _v,_J,_d = scipy.optimize.fmin_l_bfgs_b(logRegObj, numpy.zeros(M*K + K), approx_grad = True)
        _w = _v[0:M]
        _b = _v[-1]

        scores = _w*DTE + _b - numpy.log(pi/(1-pi))

        return scores.reshape((scores.shape[1],))
    


    def train_SVM_linear(self, DTR, LTR, DTE, LTE, C = 1, pi = None):

        Z = numpy.zeros(LTR.shape)
        Z[ LTR == 1] = 1
        Z[ LTR == 0] = -1

        
        DTREXT = numpy.vstack([DTR, numpy.ones((1, DTR.shape[1]))]) 

        H = numpy.dot(DTREXT.T, DTREXT)
        H = mcol(Z)*mrow(Z)*H
        empP = (LTR == 1).sum()/len(LTR)

        bounds = numpy.array([(0, C)] * LTR.shape[0])

        # balancing
        if pi != None:
            bounds[LTR == 1] = (0, C*pi/empP)
            bounds[LTR == 0] = (0, C*(1 - pi)/(1 - empP))


        def JDual(alpha):
            Ha = numpy.dot(H, mcol(alpha))
            aHa = numpy.dot(mrow(alpha), Ha)
            a1 = alpha.sum()

            return -0.5* aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

        def LDual(alpha):
            loss, grad = JDual(alpha)
            return -loss, -grad

        def JPrimal(w):
            S = numpy.dot(mrow(w), DTREXT)
            loss = numpy.maximum(numpy.zeros(S.shape), 1 -Z*S).sum()
            return -0.5*numpy.linalg.norm(w)**2 + C*loss
        
        def evaluation_SVM_linear(DTE, LTE, wStar):
        
            w = wStar[0:-1]
            b = wStar[-1]
            scores = numpy.dot(mrow(w), DTE) + b
            Labels = numpy.int32(scores > 0)
            err = (Labels != LTE).sum()
            err = err / Labels.shape[1] * 100
            return err, scores
        

        alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
            LDual,
            numpy.zeros(DTR.shape[1]),
            bounds = bounds
            # ,
            # factr = 0.0,
            # maxiter = 100000,
            # maxfun = 100000,
        )


        wStar = numpy.dot(DTREXT, mcol(alphaStar) * mcol(Z))

        err, scores = evaluation_SVM_linear(DTE, LTE, wStar)

        return err, numpy.reshape(scores,(scores.shape[1],))


    
 
    def train_SVM_kernel(self, DTR, LTR, DTE, LTE, C = 1, K = 0, gamma = 1, d = 2, c = 1, RBF = False, pi = None):
        
        Z = numpy.zeros(LTR.shape)
        Z[ LTR == 1] = 1
        Z[ LTR == 0] = -1

        #Radial Basis Function
        if RBF:
            Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2*numpy.dot(DTR.T, DTR)
            kernel = numpy.exp(-gamma*Dist) + K
        #Polinomial
        else:
            kernel = (numpy.dot(DTR.T, DTR) + c)**d
            
        H = mcol(Z)*mrow(Z)*kernel

        empP = (LTR == 1).sum() / len(LTR)
        bounds = numpy.array([(0, C)] * LTR.shape[0])

        # calibration
        if pi != None:
            bounds[LTR == 1] = (0, C * pi / empP)
            bounds[LTR == 0] = (0, C * (1 - pi) / (1 - empP))

        def JDual(alpha):
            Ha = numpy.dot(H, mcol(alpha))
            aHa = numpy.dot(mrow(alpha), Ha)
            a1 = alpha.sum()

            return -0.5* aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

        def LDual(alpha):
            loss, grad = JDual(alpha)
            return -loss, -grad

        
        def evaluation_SVM_kernel(s):
            label = numpy.int32(s > 0)
            err = (label != LTE).sum()
            err = err / LTE.shape[0] * 100
            return err
        

        alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
            LDual,
            numpy.zeros(DTR.shape[1]),
            bounds = bounds #,
            # factr = 0.0,
            # maxiter = 100000,
            # maxfun = 100000,
        )

        
        s = numpy.zeros(DTE.shape[1])

        #Radial Basis Function
        if RBF:
            for i in range(0, DTE.shape[1]):
                for j in range(0, DTR.shape[1]):
                    exp = numpy.linalg.norm(DTR[:, j] - DTE[:, i]) ** 2 * gamma
                    kern = numpy.exp(-exp) + K
                    s[i] += alphaStar[j] * Z[j] * kern
        #Polinomial
        else:
            for i in range(0, DTE.shape[1]):
                for j in range(0, DTR.shape[1]):

                    kern = (numpy.dot(DTR[:, j], DTE[:, i]) + c)**d
                    s[i] += alphaStar[j] * Z[j] * kern


        err = evaluation_SVM_kernel(s)
        return err, s

    def computeGMM(self, DTR, LTR, DTE, LTE, alpha, n_component, psi = 0.01, type_covariance = 'Full'):

        n_class = 2
        #n_class = LTR.max() + 1
        
        log_dens = []
        
        for i in range(n_class):
            DTR_i = DTR[:, LTR == i]
            gmm_ci = GMM_LBG(DTR_i, alpha, n_component, psi, type_covariance)
            _, logDensity_ci = GMM_ll_perSample(DTE, gmm_ci)
            log_dens.append(logDensity_ci)
        
        
        
        log_dens = numpy.vstack(log_dens)
        llr = log_dens[1, :] - log_dens[0, :]
        
        
        predictions = numpy.argmax(log_dens, axis=0)
        correct = LTE == predictions
        err = sum(correct)/len(LTE)


        return llr
          
    


