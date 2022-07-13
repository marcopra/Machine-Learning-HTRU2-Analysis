from turtle import color
import numpy
import scipy
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
from Evaluation import *
import seaborn as sns

M = 8 #number of features

def plot_scatter2d(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    
    plt.figure()
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    
    plt.scatter(D0[0, :], D0[1, :], label = '0')
    plt.scatter(D1[0, :], D1[1, :], label = '1')
    

    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    #plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
    plt.show()

def plot_minDCF(x1, y1, x2, y2, x3, y3, xlabel, title, g):
    
    plt.figure()
    plt.title(f'{title}')
    plt.xlabel(f'{xlabel}')
    plt.ylabel("minDCF")

    
    plt.plot(x1, y1, label = 'minDCF(π = 0.5)', color = 'r', linewidth=2.0)
    plt.plot(x2, y2, label = 'minDCF(π = 0.9)', color = 'b', linewidth=2.0)
    plt.plot(x3, y3, label = 'minDCF(π = 0.1)', color = 'g', linewidth=2.0)
    plt.legend(loc = 'best')
    plt.xscale('log')
    plt.xlim([1e-5,1e5])
    #plt.savefig(f'minDCF_{title}')
    plt.savefig(f'minDCF_RBF_{g}.png')
    #plt.show()

    
def plot_scatter3d(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.scatter(D0[0, :], D0[1, :], D0[2, :], label = '0')
    ax.scatter(D1[0, :], D1[1, :], D1[2, :], label = '1')
    

    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    #plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
    plt.show()

def plot_features(D, L, filename = ""):
    D0 = D[:, L == 0]
    D1 = D[: ,L == 1]

    for i in range(D.shape[0]):
        figure = plt.figure()
        plt.title(f'feature_{i}')
        plt.hist(D0[i, :], bins=100, density=True, alpha=0.6, label = 'negative', color='b', edgecolor = 'black')
        plt.hist(D1[i, :], bins=100, density=True, alpha=0.6, label = 'positive', color='r', edgecolor = 'black')
        plt.legend(loc = 'best')
        plt.savefig(f'{filename}feature_{i}.png')
        plt.close(figure)
    

def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))

def load(fname):
    DList = []
    labelsList = []
    
    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:M]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0) 
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1]) 
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def heat_map_plot(D):
    corr = numpy.zeros((D.shape[0], D.shape[0]))
    for i in range  (D.shape[0]):
        for j in range(D.shape[0]):
            corr[i, j] = numpy.cov(D[i, :], D[j, :]) [0][1] / (numpy.std(D[i, :] * numpy.std(D[j, :])))
    plt.figure(figsize=(12,7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues',mask=numpy.triu(corr,+1))
    plt.savefig("heatmap0.png")
    plt.show()
    

def compute_empirical_cov(X):
    mu = compute_empirical_mean(X)
    cov = numpy.dot((X - mu), (X - mu).T)/X.shape[1]
    return cov

def compute_empirical_mean(X):
    return mcol(X.mean(1))

#compute the between class covariance matrix
def compute_sb(X, L):
    SB = 0
    muG = compute_empirical_mean(X)
    for i in set(list(L)):
        D = X[:, L==i]
        mu = compute_empirical_mean(D)

        SB += D.shape[1]*numpy.dot((mu - muG), (mu - muG).T) 
    return SB/ X.shape[1]

#compute the within class covariance matrix
def compute_sw(D,L):
    SW = 0
    
    for i in [0,1]:
        SW += (L==i).sum() * compute_empirical_cov(D[:, L== i])

    SW = SW/D.shape[1]
    return SW


def logpdf_GAU_ND(X, mu, C):

    P = numpy.linalg.inv(C)
    const= -0.5 * X.shape[0] *numpy.log(2*numpy.pi) 
    const += - 0.5*numpy.linalg.slogdet(C)[1]
    
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const -0.5 * numpy.dot((x - mu).T , numpy.dot(P , (x - mu)))
        Y.append(res)
    

    return numpy.array(Y).ravel()


def GMM_LBG(X, alpha, nComponents, psi = 0.01, type_covariance = 'Full'):
    gmm = [(1, compute_empirical_mean(X), compute_empirical_cov(X))]
    
    while len(gmm) <= nComponents:
        
        
        gmm = GMM_EM(X, gmm, psi, type_covariance)
                
        if len(gmm) == nComponents:
            break
        
        newGmm = []
        for i in range(len(gmm)):
            (w, mu, sigma) = gmm[i]
            U, s, Vh = numpy.linalg.svd(sigma)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            newGmm.append((w/2, mu + d, sigma))
            newGmm.append((w/2, mu - d, sigma))
        gmm = newGmm
            
    return gmm

def GMM_ll_perSample(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = numpy.zeros((G,N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
    
    return  S, scipy.special.logsumexp(S, axis = 0)


def GMM_EM(X, gmm, psi = 0.01, type_covariance = 'Full'):
    

    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    D = X.shape[0]


    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        
        SJ = numpy.zeros((G,N))
        for g in range(G):
            SJ[g,:] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2])  + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis = 0) #marginal density

        llNew = SM.sum()/N #loglikelihood averaged
        P = numpy.exp(SJ - SM)

        if type_covariance == 'Full':

            gmmNew = []
        
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (mrow(gamma)*X).sum(1)
                S = numpy.dot(X,  (mrow(gamma)*X).T)
                w = Z/N
                mu = mcol(F/Z)
                Sigma = S/Z - numpy.dot(mu, mu.T)
                U, s, _ = numpy.linalg.svd(Sigma)
                s[s<psi] = psi
                Sigma = numpy.dot(U, mcol(s)*U.T)
                gmmNew.append((w, mu, Sigma))
            gmm = gmmNew

        elif type_covariance == 'Tied':

            gmmNew = []
            tiedSigma = numpy.zeros((D, D))
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (mrow(gamma)*X).sum(1)
                S = numpy.dot(X,  (mrow(gamma)*X).T)
                w = Z/N
                mu = mcol(F/Z)
                Sigma = S/Z - numpy.dot(mu, mu.T)
                tiedSigma += Z * Sigma
                gmmNew.append((w, mu))
            gmm = gmmNew
            tiedSigma = tiedSigma / N
            U, s, _ = numpy.linalg.svd(tiedSigma)
            s[s<psi] = psi
            tiedSigma = numpy.dot(U, mcol(s)*U.T)

            gmmNew = []
            for i in range(G):
                (w, mu) = gmm[i]
                gmmNew.append((w, mu, tiedSigma))
            gmm = gmmNew
        
        elif type_covariance == 'Diag':
            gmmNew = []
        
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (mrow(gamma)*X).sum(1)
                S = numpy.dot(X,  (mrow(gamma)*X).T)
                w = Z/P.sum()
                mu = mcol(F/Z)
                Sigma = S/Z - numpy.dot(mu, mu.T)
                Sigma *= numpy.eye(Sigma.shape[0])
                U, s, _ = numpy.linalg.svd(Sigma)
                s[s<psi] = psi
                Sigma = numpy.dot(U, mcol(s)*U.T)
                gmmNew.append((w, mu, Sigma))
            gmm = gmmNew
        
        elif type_covariance =='Tied-Diag':
            gmmNew = []
            tiedSigma = numpy.zeros((D, D))
        
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (mrow(gamma)*X).sum(1)
                S = numpy.dot(X,  (mrow(gamma)*X).T)
                w = Z/P.sum()
                mu = mcol(F/Z)
                Sigma = S/Z - numpy.dot(mu, mu.T)
                Sigma *= numpy.eye(Sigma.shape[0])
                tiedSigma += Z * Sigma
                gmmNew.append((w, mu))
            gmm = gmmNew
            tiedSigma = tiedSigma / N
            tiedSigma *= numpy.eye(Sigma.shape[0])
            U, s, _ = numpy.linalg.svd(tiedSigma)
            s[s<psi] = psi
            tiedSigma = numpy.dot(U, mcol(s)*U.T)

            gmmNew = []
            for i in range(G):
                (w, mu) = gmm[i]
                gmmNew.append((w, mu, tiedSigma))
            gmm = gmmNew

    #     print("A0",llNew)
    # print("Diff: ", llNew - llOld)
    # print("Avg", llNew)
    return gmm

def my_shuffle(D, L, seed = 0):

    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1]) 
    D = D[:, idx]
    L = L[idx]
    return D,L

def score_shuffle_and_split(scores, L):
    numpy.random.seed(0)
    idx = numpy.random.permutation(scores.shape[0]) 
    nTrain = int(scores.shape[0]*1.0/2.0)
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    scoreTR = scores[idxTrain]
    scoreTE = scores[idxTest]
    LabelTR = L[idxTrain]
    LabelTE = L[idxTest]

    return scoreTR, LabelTR, scoreTE, LabelTE

def kfold_split(D, L, k = 10):
    D_split = []
    L_split = []
    
    for i in range(0, k):
        if i == k-1:
            D_split.append(D[:, int(i/k*D.shape[1]):])
            L_split.append(L[int(i/k*D.shape[1]):])
        else:
            D_split.append(D[:, int(i/k*D.shape[1]):int((i+1)/k*D.shape[1])])
            L_split.append(L[int(i/k*D.shape[1]):int((i+1)/k*D.shape[1])])
    return D_split, L_split

def kfold_validation(D, L, to_eval = 0, k = 10):

    DTR = numpy.empty(shape=[D.shape[0], 0])
    LTR = numpy.empty(shape = 0)
    DTE = numpy.empty(shape=[D.shape[0], 0])
    LTE = numpy.empty(shape = 0)

    D_kfold, L_kfold = kfold_split(D, L, k)
    
    for j in range (0, k):
        to_add_data = numpy.array(D_kfold[j])
        to_add_label = numpy.array(L_kfold[j])
        if j != to_eval:
            DTR = numpy.hstack((DTR, to_add_data))
            LTR = numpy.hstack((LTR, to_add_label))
    
        else :
            DTE = numpy.hstack((DTE, to_add_data))
            LTE = numpy.hstack((LTE, to_add_label))
    

    return DTR, LTR, DTE, LTE


#--------------CLASSIFIERS PLOT------------------------#

def plot_minDCF_RBF(classifier, preprocessing, D, L):

    #Setting the parameters
    m_PCA = [None, 7]
    gamma  = [1e-5, 1e-4, 1e-3]
    p, cfp, cfn = (0.5, 1, 1)
    C = numpy.logspace(-4, -1, 10)
    k = 5  

   
    for j in range(len(m_PCA)):
        m = m_PCA[j]
        g1 = []
        g2 = []
        g3 = []
        for g in gamma:
            for c_ in C:
                
                scores = numpy.array([])
                for i in range(0,k):
                    DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                    if m != 8:

                        P = preprocessing.compute_pca(DTR, m)
                        DTR = numpy.dot(P.T, DTR)
                        DTE = numpy.dot(P.T, DTE)
                

                    acc,llr =  classifier.train_SVM_kernel(DTR, LTR, DTE, LTE, gamma = g , C = c_, RBF = True)
    
                    scores = numpy.concatenate((scores, llr))
                
                minDCF = compute_min_DCF(scores, L, p, cfn, cfp)
                
                if g == 1e-5:
                    g1.append(minDCF)
                elif g == 1e-4:
                    g2.append(minDCF)
                elif g == 1e-3:
                    g3.append(minDCF)
                
                print ("{:<20} {:<25} ".format(f'RBF_SVM(c = {c_}, gamma = {g})',minDCF))
        
        figure = plt.figure()
        plt.xlabel("C")
        plt.ylabel("minDCF")
        plt.plot(C, g1, label = 'minDCF(π = 0.5) - logγ= -5', color = 'r', linewidth=2.0)
        plt.plot(C, g2,  label = 'minDCF(π = 0.5) - logγ= -4', color = 'b', linewidth=2.0)
        plt.plot(C, g3,  label = 'minDCF(π = 0.5) - logγ= -3', color = 'g', linewidth=2.0)
        plt.legend(loc = 'best')
        plt.xscale('log')
        plt.xlim([1e-4, 1e-1])

        plt.savefig(f'RBF_SVM_k_{k}_m_{m}.png')

def plot_minDCF_poly(classifier, preprocessing, D, L):

    #Setting the parameters
    m_PCA = [None, 7]
    p, cfp, cfn = (0.5, 1, 1)
    C = numpy.logspace(-4, 0, 10)
    c_small = [10, 1, 0, 20]
    k = 5  
    C = [0.001]
   
    for j in range(len(m_PCA)):
        m = m_PCA[j]
        c1 = []
        c2 = []
        c3 = []
        c4 = []
        for c_small_ in c_small:
            for c_ in C:
                
                scores = numpy.array([])
                for i in range(0,k):
                    DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                    if m is not None:

                        P = preprocessing.compute_pca(DTR, m)
                        DTR = numpy.dot(P.T, DTR)
                        DTE = numpy.dot(P.T, DTE)
                

                    acc,llr =  classifier.train_SVM_kernel(DTR, LTR, DTE, LTE, C = c_, K = 1, d = 2, c = c_small_)
    
                    scores = numpy.concatenate((scores, llr))
                
                minDCF = compute_min_DCF(scores, L, p, cfn, cfp)
                
                if c_small_ == 0:
                    c1.append(minDCF)
                elif c_small_ == 1:
                    c2.append(minDCF)
                elif c_small_ == 10:
                    c3.append(minDCF)
                elif c_small_ == 20:
                    c4.append(minDCF)
                
                print ("{:<10} {:<25} ".format(f'polySVM(C = {c_}, c_small = {c_small_})',minDCF))
        
        figure = plt.figure()
        plt.xlabel("C")
        plt.ylabel("minDCF")
        plt.plot(C, c1, label = 'minDCF(π = 0.5) - c = 0', color = 'r', linewidth=2.0)
        plt.plot(C, c2,  label = 'minDCF(π = 0.5) - c = 1', color = 'b', linewidth=2.0)
        plt.plot(C, c3,  label = 'minDCF(π = 0.5) - c = 10', color = 'g', linewidth=2.0)
        plt.plot(C, c4,  label = 'minDCF(π = 0.5) - c = 20', color = 'y', linewidth=2.0)
        plt.legend(loc = 'best')
        plt.xscale('log')
        plt.xlim([1e-4, 1])

        plt.savefig(f'RBF_SVM_k_{k}_m_{m}.png')

def plot_minDCF_linearSVM(classifier, preprocessing, D, L):

    #Setting the parameters
    m_PCA = [None, 7]
    prob_cost = [[0.5,1,1], [0.9, 1, 1], [0.1, 1, 1]]
    
    C = numpy.logspace(-4, -1, 10)
    C = [0.1]
    pi_cal = [None, 0.5, 0.9, 0.1]

    k = 5    

    for j in range(len(m_PCA)):
        m = m_PCA[j]
        for pit in pi_cal:
            x5 = []
            x9 = []
            x1 = []
            for p, cfn, cfp in prob_cost:
                for c_ in C:

                    scores = numpy.array([])
                    for i in range(0,k):
                        DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                        if m is not None:

                            P = preprocessing.compute_pca(DTR, m)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)


                        acc,llr =  classifier.train_SVM_linear(DTR, LTR, DTE, LTE, c_, pi = pit)

                        scores = numpy.concatenate((scores, llr))

                    minDCF = compute_min_DCF(scores, L, p, cfn, cfp)

                    if p == 0.5:
                        x5.append(minDCF)
                    elif p == 0.9:
                        x9.append(minDCF)
                    elif p == 0.1:
                        x1.append(minDCF)

                    print ("{:<10} {:<25} {:<25} ".format(f'linearSVM(c = {c_}, pit = {pit})',minDCF, p))



            figure = plt.figure()
            plt.xlabel("C")
            plt.ylabel("minDCF")
            plt.plot(C, x5, label = 'minDCF(π = 0.5)', color = 'r', linewidth=2.0)
            plt.plot(C, x9,  label = 'minDCF(π = 0.9)', color = 'b', linewidth=2.0)
            plt.plot(C, x1,  label = 'minDCF(π = 0.1)', color = 'g', linewidth=2.0)
            plt.legend(loc = 'best')
            plt.xscale('log')
            plt.xlim([1e-4, 1e-1])

            plt.savefig(f'linearSVM_pit_{pit}_m_{m}.png')



def plot_minDCF_logReg(classifier, preprocessing, D,L, quadratic = False):

    #Setting the parameters
    m_PCA = [None, 7]
    prob_cost = [[0.5,1,1], [0.9, 1, 1], [0.1, 1, 1]]
    lamb = numpy.logspace(-5, 3, 12)
    kfold = [None, 5]
    

    for t in range(len(kfold)):
        k = kfold[t]
        for j in range(len(m_PCA)):
            m = m_PCA[j]
            x5 = []
            x9 = []
            x1 = []
            for p, cfn, cfp in prob_cost:
                for l in lamb:
                    
                    if k is None:
                        (DTR,LTR),(DTE,LTE) = split_db_2to1(D, L)
                        if m != 8:

                            P = preprocessing.compute_pca(DTR, m)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)
                        
                        if quadratic: 
                            DTR = preprocessing.feature_expansion(DTR)
                            DTE = preprocessing.feature_expansion(DTE)

                        acc,llr =  classifier.compute_logreg(DTR,LTR,DTE,LTE, l)
        
                        minDCF = compute_min_DCF(llr, LTE, p, cfn, cfp)
                       
                        if p == 0.5:
                            x5.append(minDCF)
                        elif p == 0.9:
                            x9.append(minDCF)
                        elif p == 0.1:
                            x1.append(minDCF)
                    else:
                        scores = numpy.array([])
                        for i in range(0,k):
                            DTR,LTR,DTE,LTE = kfold_validation(D, L, i, k)
                            if m != 8:

                                P = preprocessing.compute_pca(DTR, m)
                                DTR = numpy.dot(P.T, DTR)
                                DTE = numpy.dot(P.T, DTE)
                            
                            if quadratic: 
                                DTR = preprocessing.feature_expansion(DTR)
                                DTE = preprocessing.feature_expansion(DTE)
    
                            acc,llr =  classifier.compute_logreg(DTR,LTR,DTE,LTE, l)
            
                            scores = numpy.concatenate((scores, llr))
                        
                        minDCF = compute_min_DCF(scores, L, p, cfn, cfp)
                        
                        if p == 0.5:
                            x5.append(minDCF)
                        elif p == 0.9:
                            x9.append(minDCF)
                        elif p == 0.1:
                            x1.append(minDCF)
                        
                    print ("{:<10} {:<25} ".format(f'LogReg(l = {l})',minDCF))
            
            figure = plt.figure()
            plt.xlabel("λ")
            plt.ylabel("minDCF")
            plt.plot(lamb, x5, label = 'minDCF(π = 0.5)', color = 'r', linewidth=2.0)
            plt.plot(lamb, x9,  label = 'minDCF(π = 0.9)', color = 'b', linewidth=2.0)
            plt.plot(lamb, x1,  label = 'minDCF(π = 0.1)', color = 'g', linewidth=2.0)
            plt.legend(loc = 'best')
            plt.xscale('log')
            plt.xlim([1e-5, 1e3])

            if quadratic:
                plt.savefig(f'quadraticLogReg_k_{k}_m_{m}.png')

            else:
                plt.savefig(f'linearLogReg_k_{k}_m_{m}.png')

            


def plot_minDCF_GMM(classifier, preprocessing, D,L):
    m_PCA = [None, 7]
    prob_cost = [[0.5, 1, 1], [0.9, 1, 1], [0.1, 1, 1]]
    components = [2, 4, 8, 16, 32]
    types = ['Full', 'Tied', 'Diag']
    kfold = [None, 5]
    p = [0.5, 0.9, 0.1]


    for t in range(len(kfold)):
        k = kfold[t]
        for j in range(len(m_PCA)):
            for type in types:
                m = m_PCA[j]
                x5 = []
                x9 = []
                x1 = []
                for p, cfn, cfp in prob_cost:
                    for component in components:
                        if k is None:
                            (DTR,LTR),(DTE,LTE) = split_db_2to1(D, L)
                            if m is not None:

                                P = preprocessing.compute_pca(DTR, m)
                                DTR = numpy.dot(P.T, DTR)
                                DTE = numpy.dot(P.T, DTE)

                            llr = classifier.computeGMM(DTR, LTR, DTE, LTE, 0.1 , component, 0.1, type)
            
                            minDCF = compute_min_DCF(llr, LTE, p, cfn, cfp)
                        
                            if p == 0.5:
                                x5.append(minDCF)
                            elif p == 0.9:
                                x9.append(minDCF)
                            elif p == 0.1:
                                x1.append(minDCF)
                        else:
                            scores = numpy.array([])
                            for i in range(0, k):
                                DTR, LTR, DTE, LTE = kfold_validation(D, L, i, k)
                                if m is not None:
                                    P = preprocessing.compute_pca(DTR, m)
                                    DTR = numpy.dot(P.T, DTR)
                                    DTE = numpy.dot(P.T, DTE)


                                llr = classifier.computeGMM(DTR, LTR, DTE, LTE, 0.1 , component, 0.1, type)

                                scores = numpy.concatenate((scores, llr))

                            minDCF = compute_min_DCF(scores, L, p, cfn, cfp)

                            if p == 0.5:
                                x5.append(minDCF)
                            elif p == 0.9:
                                x9.append(minDCF)
                            elif p == 0.1:
                                x1.append(minDCF)

                        print("{:<10} {:<25} ".format(f'GMM(n = {component}, type = {type})', minDCF))

                fig, ax = plt.subplots(constrained_layout = True)
                ax.set_xlabel("n° components")
                ax.set_ylabel("minDCF")
                ax.plot(components, x5, label='minDCF(π = 0.5)', color='r', linewidth=2.0)
                ax.plot(components, x9, label='minDCF(π = 0.9)', color='b', linewidth=2.0)
                ax.plot(components, x1, label='minDCF(π = 0.1)', color='g', linewidth=2.0)
                ax.legend(loc='best')
                ax.set_xscale('log', basex = 2)
                ax.set_xlim([2, 32])

        
                plt.savefig(f'GMM_m_{m}_t_{type}.png')
    
def plotGMM_bar(classifier, preprocessing, D,L):
    components = [2, 4, 8, 16, 32, 64]

    type_covariances = ['Full', 'Tied', 'Diag', 'Tied-Diag']
    prob_cost = [[0.5, 1, 1]]
    k = 2
   
    preproc =  [None, 'Gaussianization', 'PCA']
    minDcfs = numpy.zeros((4, len(components)))
    minDcfsG = numpy.zeros((4, len(components)))
    minDcfsP = numpy.zeros((4, len(components)))
    
    for p, cfn, cfp in prob_cost:
        for ind_comp ,c in enumerate(components):
            for type_prepocessing in preproc:
                for ind_cov, type_covariance in enumerate(type_covariances):
                    scores = numpy.array([])
                    for i in range(0, k):

                        DTR, LTR, DTE, LTE = kfold_validation(D, L, i, k)
                        

                        if type_prepocessing == 'Gaussianization':
                            DTR, DTE = preprocessing.gaussianization(DTR, DTE)

                        elif type_prepocessing == 'PCA':
                            P = preprocessing.compute_pca(DTR, 7)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)


                        llr = classifier.computeGMM(DTR, LTR, DTE, LTE, 0.01 , c, 0.01, type_covariance)


                        scores = numpy.concatenate((scores, llr))


                    minDCF = compute_min_DCF(scores, L, p, cfn, cfp)

                    if type_prepocessing == 'Gaussianization':
                        minDcfsG[ind_cov, ind_comp] = minDCF
                    elif type_prepocessing == 'PCA':
                        minDcfsP[ind_cov, ind_comp] = minDCF
                    else:
                        minDcfs[ind_cov, ind_comp] = minDCF
                    



                    print("{:<35} {:<25} {:<8} ".format(f'GMM(n = {c}, type = {type_covariance})', minDCF, p))


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout = True)
    width = 0.25
    ind = numpy.arange(len(components)) 
    b1n = ax1.bar(ind, minDcfs[0], width, color = 'r')
    b1g = ax1.bar(ind+width, minDcfsG[0], width, color='g')
    b1p = ax1.bar(ind+width+width, minDcfsP[0], width, color='b')
    ax1.legend((b1n, b1g, b1p), ('Raw', 'Gaussianized', 'PCA'))
    ax1.title.set_text('Full')
    ax1.set_xticks(ind + width, components)
    b2n = ax2.bar(ind, minDcfs[1], width, color = 'r')
    b2g = ax2.bar(ind+width, minDcfsG[1], width, color='g')
    b2p = ax2.bar(ind+width+width, minDcfsP[1], width, color='b')
    ax2.legend((b2n, b2g, b2p), ('Raw', 'Gaussianized', 'PCA'))
    ax2.title.set_text('Diag')
    ax2.set_xticks(ind + width, components)
    b3n = ax3.bar(ind, minDcfs[2], width, color = 'r')
    b3g = ax3.bar(ind+width, minDcfsG[2], width, color='g')
    b3p = ax3.bar(ind+width+width, minDcfsP[2], width, color='b')
    ax3.legend((b3n, b3g, b3p), ('Raw', 'Gaussianized', 'PCA'))
    ax3.title.set_text('Tied')
    ax3.set_xticks(ind + width, components)
    b4n = ax4.bar(ind, minDcfs[3], width, color = 'r')
    b4g = ax4.bar(ind+width, minDcfsG[3], width, color='g')
    b4p = ax4.bar(ind+width+width, minDcfsP[3], width, color='b')
    ax4.legend((b4n, b4g, b4p), ('Raw', 'Gaussianized', 'PCA'))
    ax4.title.set_text('Tied Diag')
    ax4.set_xticks(ind + width, components)
    plt.savefig('GMM_bars.png')


def error_plot(scores, Labels, filename = "none", pi = None, calibrated = False):

    p = numpy.linspace(-3, 3, 21)
    figure = plt.figure()
    plt.xlabel("prior")
    plt.ylabel("minDCF")
    plt.plot(p, bayes_error_plot(p, scores, Labels, minCost= False), label= f'{filename} - actDCF', color = 'r')
    plt.plot(p, bayes_error_plot(p, scores, Labels, minCost= True), linestyle='dashed', label= f'{filename} - minDCF' , color = 'b')
    plt.legend(loc='best')
    if calibrated:
        plt.savefig(f'calibrated_{filename}_error_plot_p_{pi}.png')
    else:
        plt.savefig(f'{filename}_error_plot_p_{pi}.png')