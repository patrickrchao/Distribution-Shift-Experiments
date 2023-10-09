import numpy as np 

class Estimator:
    def __call__(self, data):
        return self._estimate(data)
    
    def _estimate(self,data):
        """
        data : np.array
            a 3D array of shape (num_copies, n, p) containing num_copies copies of the data.
        Returns
        -------
        np.array
            A 2D array of shape (num_copies,p) containing the estimated parameter.
        """
        pass

class MeanEstimator(Estimator):
    name = "Mean"
    def _estimate(self, data):
        assert len(data.shape) == 3
        return np.mean(data, axis=1)

class MedianEstimator(Estimator):
    name = "Median"
    def _estimate(self, data):
        assert len(data.shape) == 3
        return np.median(data, axis=1)

class kthEstimator(Estimator):
    def __init__(self, k):
        self.k = k
        self.name = f"{k}-Estimator"
    def _estimate(self, data):
        assert len(data.shape) == 3
        assert data.shape[2] == 1
        sorted = np.sort(data, axis=1)
        n = data.shape[1]
        return (sorted[:,self.k,:] + sorted[:,(n-self.k+1),:])/2

class MaxMinEstimator(Estimator):
    name = "MaxMin"
    def _estimate(self, data):
        assert len(data.shape) == 3
        return (np.max(data, axis=1) + np.min(data, axis=1))/2
        
        
class LeastSquaresEstimator(Estimator):
    name = "LeastSquares"
    def __init__(self, X):
        self.X = X
    
    def _estimate(self, Y):
        """
        Y : np.array
            a 3D array of shape (num_copies, 1, n) containing num_copies copies of the data.
        Returns
        -------
        np.array
            A 2D array of shape (num_copies,p) containing the estimated parameter
        """
        X = self.X

        assert(len(Y.shape) == 3 and Y.shape[1] == 1)
        assert(len(X.shape) == 2)
    
        
        # Swap Y's first two axes
        estimate = np.linalg.inv(X.T @ X) @ X.T @ (Y.squeeze().T)
        return estimate.T


class GeneralizedLeastSquaresEstimator(Estimator):
    name = "GeneralizedLeastSquares"
    def __init__(self, X,Sigma):
        self.X = X
        self.Sigma = Sigma
    
    def _estimate(self, Y):
        """
        Y : np.array
            a 3D array of shape (num_copies, 1, n) containing num_copies copies of the data.
        Returns
        -------
        np.array
            A 2D array of shape (num_copies,p) containing the estimated parameter
        """
        X = self.X
        Sigma = self.Sigma
        Sigma_inv = np.linalg.inv(Sigma)
        assert(len(Y.shape) == 3 and Y.shape[1] == 1)
        assert(len(X.shape) == 2)
        
        # Swap Y's first two axes
        estimate = np.linalg.inv(X.T @ Sigma_inv @ X) @ X.T @ Sigma_inv @ (Y.squeeze().T)
        return estimate.T


        
        
        
        
        
        
