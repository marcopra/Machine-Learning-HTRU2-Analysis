import numpy
import pylab
import matplotlib.pyplot as plt

def assign_labels(scores, pi = 0.5, Cfn = 1, Cfp = 1, th = None):
    if th is None:
        th = -numpy.log(pi * Cfn) + numpy.log((1 - pi)*Cfp)
    P = scores > th
    return numpy.int32(P)

def compute_conf_matrix_binary(Pred, Labels):
    C = numpy.zeros((2,2))
    C[0,0] = ((Pred == 0) * (Labels == 0)).sum()
    C[0,1] = ((Pred == 0) * (Labels == 1)).sum()
    C[1,0] = ((Pred == 1) * (Labels == 0)).sum()
    C[1,1] = ((Pred == 1) * (Labels == 1)).sum()
    return C

def compute_emp_Bayes_binary(CM, pi, Cfn, Cfp): #DCF_u
    
    fnr = CM[0,1]/(CM[0,1] + CM[1,1])
    fpr = CM[1,0]/(CM[0,0] + CM[1,0])
    return pi * Cfn * fnr + (1 - pi) * Cfp * fpr

def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp): #DCF
    empBayes = compute_emp_Bayes_binary(CM, pi, Cfn, Cfp)
    return empBayes/ min(pi*Cfn, (1-pi)*Cfp)

def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th = None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th = th)
    CM = compute_conf_matrix_binary(Pred, labels)
    return compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp)


def compute_min_DCF(scores, labels, pi, Cfn, Cfp):

    t = numpy.array(scores)
    t.sort()

    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th = _th))
    return numpy.array(dcfList).min()


def bayes_error_plot(pArray, scores, labels, minCost = False):
    y = []
    for p in pArray:
        pi = 1.0/ (1.0 + numpy.exp(-p))
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1))   
    return numpy.array(y)

def computeFPR_FPR_TPR(C):
    # Compute FNR and FPR
    FNR = C[0][1]/(C[0][1]+C[1][1])
    TPR = 1-FNR
    FPR = C[1][0]/(C[0][0]+C[1][0])
    return FNR, FPR, TPR

def plotROC(scores1, scores2, scores3, L1, L2 , L3):

    t = numpy.array(scores1)
    t.sort()
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    FPR = numpy.zeros(t.size)
    TPR = numpy.zeros(t.size)
    FNR = numpy.zeros(t.size)
    for idx,t_ in enumerate(t):
        C = compute_conf_matrix_binary(assign_labels(scores1, th = t_), L1)
        FNR[idx], FPR[idx], TPR[idx] = computeFPR_FPR_TPR(C)

    #----------------------------------------
    t = numpy.array(scores2)
    t.sort()
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    FPR2 = numpy.zeros(t.size)
    TPR2 = numpy.zeros(t.size)
    FNR2 = numpy.zeros(t.size)
    for idx,t_ in enumerate(t):
        C = compute_conf_matrix_binary(assign_labels(scores2, th = t_), L2)
        FNR2[idx], FPR2[idx], TPR2[idx] = computeFPR_FPR_TPR(C)
    
    #----------------------------------------
    t = numpy.array(scores3)
    t.sort()
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    FPR3 = numpy.zeros(t.size)
    TPR3 = numpy.zeros(t.size)
    FNR3 = numpy.zeros(t.size)
    for idx,t_ in enumerate(t):
        C = compute_conf_matrix_binary(assign_labels(scores3, th = t_), L3)
        FNR3[idx], FPR3[idx], TPR3[idx] = computeFPR_FPR_TPR(C)

    # Function used to plot TPR(FPR)
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(FPR, TPR, linewidth=2, color='r')
    plt.plot(FPR2, TPR2, linewidth=2, color='b')
    plt.plot(FPR3, TPR3, linewidth=2, color='g')
    plt.legend(["Tied-Cov", "Linear Log Reg", "Linear SVM"], loc='best')

    # plt.legend(["Tied-Cov", "Quad Log Reg", "Poly SVM"], loc='best')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig('ROC_Test.png')
    return

def plotDET(scores1, scores2, scores3, L1, L2, L3):

    t = numpy.array(scores1)
    t.sort()
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    FPR = numpy.zeros(t.size)
    TPR = numpy.zeros(t.size)
    FNR = numpy.zeros(t.size)
    for idx,t_ in enumerate(t):
        C = compute_conf_matrix_binary(assign_labels(scores1, th = t_), L1)
        FNR[idx], FPR[idx], TPR[idx] = computeFPR_FPR_TPR(C)

    #----------------------------------------
    t = numpy.array(scores2)
    t.sort()
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    FPR2 = numpy.zeros(t.size)
    TPR2 = numpy.zeros(t.size)
    FNR2 = numpy.zeros(t.size)
    for idx,t_ in enumerate(t):
        C = compute_conf_matrix_binary(assign_labels(scores2, th = t_), L2)
        FNR2[idx], FPR2[idx], TPR2[idx] = computeFPR_FPR_TPR(C)
    
    #----------------------------------------
    t = numpy.array(scores3)
    t.sort()
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    FPR3 = numpy.zeros(t.size)
    TPR3 = numpy.zeros(t.size)
    FNR3 = numpy.zeros(t.size)
    for idx,t_ in enumerate(t):
        C = compute_conf_matrix_binary(assign_labels(scores3, th = t_), L3)
        FNR3[idx], FPR3[idx], TPR3[idx] = computeFPR_FPR_TPR(C)

    # Function used to plot TPR(FPR)
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(FPR, FNR, linewidth=2, color='r')
    plt.plot(FPR2, FNR2, linewidth=2, color='b')
    plt.plot(FPR3, FNR3, linewidth=2, color='g')
    plt.legend(["Tied-Cov", "Linear Log Reg", "Linear SVM"], loc='best')
    # plt.legend(["Tied-Cov", "Quad Log Reg", "Poly SVM"], loc='best')
    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.xlim(1e-4, 1)
    plt.ylim(1e-4, 1)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('DET_Test.png')
    return
