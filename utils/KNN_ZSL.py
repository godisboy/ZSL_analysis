import sklearn.metrics.pairwise
import numpy as np
from numpy.linalg import inv
from scipy.misc import imrotate

def compute_W(train_features, train_attritubes):
    XtX =  np.matmul(np.transpose(train_features), train_features)
    XtS =  np.matmul(np.transpose(train_features), train_attritubes)
    W   =  np.matmul(inv(XtX + .005 * np.eye(128)), XtS)
    return W

def NormaliseRows(S_te_pro):
    ''' This function normalises the input matrix.
    S_te_pro: input matrix

    '''
    S_te_pro_T = np.transpose(S_te_pro)
    normF = np.sqrt(np.sum(np.multiply(S_te_pro_T, S_te_pro_T), axis=1))
    normFUsed = normF
    normFUsed[normFUsed == 0] = 1

    for i in range(normFUsed.shape[0]):
        S_te_pro_T[i, :] = S_te_pro_T[i, :] / normFUsed[i]

    return np.transpose(S_te_pro_T)


def knn_zsl_el(a_est_g, S_te_pro, X_te_Y, te_cl_id):
    ''' a_est_g   : [k, N] estimated semantic attribute matrix
        S_te_pro  : [c, k] prototype
        X_te_Y    : [N, 1] ground truth labels
        te_cl_id  : [c,1]  label id
        N         : the number of samples
        k         : attribute dimension size
        c         : the number of classes

        Return:
        Acc       : Accuracy
        Y_est     : Estimated labels
    '''
    dist1 = 1 - sklearn.metrics.pairwise.pairwise_distances(a_est_g, NormaliseRows(S_te_pro), 'cosine')
    Z_est = [];
    for i in range(len(dist1)):
        I = sorted(range(len(dist1[i, :])), key=lambda k: dist1[i, :][k])
        Z_est.append(te_cl_id[I[-1]])

    X_te_Y = X_te_Y[:]
    X_te_Y = np.array(X_te_Y[:])
    Z_est = np.array(Z_est)
    n = 0;
    for i in range(len(dist1)):
        if (X_te_Y[i] == Z_est[i]):
            n = n + 1;
    Acc = float(n) / len(dist1) * 100
    # print('{:0.3f}'.format (Acc))
    return Acc, Z_est
