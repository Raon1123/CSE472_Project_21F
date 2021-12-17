import numpy as np
import cupy as cp
#from cvxopt import solvers, matrix
import osqp
from scipy import sparse

"""
linear kernel for SVM
"""
def linear_kernel(sigma):
    def func(X1, X2):
        # X1 * X2^T
        ret = np.matmul(X1, X2.T)
        return ret
    return func


"""
Gaussian kernel for SVM
"""
def gaussian_kernel(sigma, cuda=False):
    def func(X1, X2):
        # Pairwise subtraction and calculate square distance
        if cuda:
            diff = cp.array(X2)[cp.newaxis,:,:] - cp.array(X1)[:, cp.newaxis, :]
            sq_dist = cp.sum(cp.square(diff), axis=-1)
            ret = cp.asnumpy(cp.exp(sq_dist / (-2.0 * cp.square(sigma))))
        else:
            diff = X2[np.newaxis,:,:] - X1[:,np.newaxis, :]
            sq_dist = np.sum(np.square(diff), axis=-1)
            ret = np.exp(sq_dist / (-2.0*np.square(sigma)))
        return ret
    return func


"""
binary SVM class
"""
class SVM():
    """
    initialize of SVM
    - kernel: kernel of support vector machine
    - C: softmargin hyperparameter
    - sigma: parameter of gaussian kernel
    """
    def __init__(self, kernel="linear", C=0.05, sigma=None):
        if kernel == "gaussian":
            self.kernel = gaussian_kernel(sigma, cuda=False)
        else:
            self.kernel = linear_kernel(sigma)
        self.C = C
        self.alpha = None

    """
    training procedure of SVM
    - X: training X (dataN, featureN)
    - y: training y (dataN,) this is binary 1 or -1
    """
    def fit(self, X, y):
        trainN = X.shape[0] # number of training data

        # change data type to double
        X_double = X.astype(np.double) 
        y_double = y.astype(np.double) 

        self.trainX = X_double
        self.trainy = y_double

        # P: (trainN, trainN)
        # P_(i,j) = y_i * y_j * x_i.T * x_j
        outer_y = np.outer(y_double, y_double) 
        P = sparse.csc_matrix(np.multiply(outer_y, self.kernel(X_double, X_double)))
        q = -1.0 * np.ones((trainN, 1))

        # constraint
        ident = sparse.identity(trainN, dtype=np.double, format='csc')
        A = sparse.vstack([ident, y_double.T]).tocsc()
        l = np.vstack([np.zeros((trainN, 1)), np.zeros((1, 1))])
        u = np.vstack([self.C * np.ones((trainN, 1)), np.zeros((1, 1))])

        # find optimal solution
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False)
        res = prob.solve()
        self.alpha = res.x
        self.b = self.calc_b()

        return self

    """
    prediction of binary SVM
    - testX: for test data
    """
    def predict(self, testX):
        trainN = self.trainX.shape[0]
        #testN = testX.shape[0]

        ker = self.kernel(self.trainX, testX)
        t_y = self.trainy.reshape(trainN, )
        coef = self.alpha * t_y

        y = ker.T @ coef + self.b
        pred = np.sign(y)
        pred = pred.ravel()

        return pred

    def calc_b(self):
        trainN = self.trainX.shape[0]
        y = self.trainy.reshape(trainN)
        ker = self.kernel(self.trainX, self.trainX)

        add = 0

        for i in range(trainN):
            sum_result = 0
            func = ker[i].reshape(trainN)
            elem = (self.alpha * y) * func
            sum_result = np.sum(elem)
            add += self.trainy[i] - sum_result
        
        b = add / trainN

        return b