import numpy as np
from numpy.linalg import solve
#from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        V = np.diag(z)
        self.w = solve(X.T@V@X, X.T@V@y)
    
class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):
        f = 0; g = 0
        for i in range(len(y)):
            expi=np.exp(X[i]@w-y[i])
            f = f + np.log(expi+1/expi)
            g = g + X[i]*(expi**2-1)/(expi**2+1)
        return f,g


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        n = X.shape[0]
        X_bias = np.hstack((np.ones((n,1)), X))
        w = solve(X_bias.T@X_bias, X_bias.T@y)
        self.w = w

    def predict(self, X):
        n = X.shape[0]
        X_bias = np.hstack((np.ones((n,1)), X))
        return X_bias@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        n = X.shape[0]
        #X_bias = np.hstack((np.ones((n,1)), X**np.arange(1,self.p+1)))
        X_bias = X**range(self.p+1)
        self.w = solve(X_bias.T@X_bias, X_bias.T@y)

    def predict(self, X):
        n = X.shape[0]
        #X_bias = np.hstack((np.ones((n,1)), X**np.arange(1,self.p+1)))
        X_bias = X**range(self.p+1)
        return X_bias@self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        ''' YOUR CODE HERE '''
        raise NotImplementedError()
