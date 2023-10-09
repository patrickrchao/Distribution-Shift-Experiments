import numpy as np 
import perturbations
from estimators import *
from abc import ABC, abstractmethod
from bounds import Bound

class StatisticalProblem:
    hierarchy = {"CDS":["CDS"], "IDS": ["CDS", "IDS"], "JDS":["CDS","IDS","JDS"]}
    def __init__(self, **params):
        """
        Parameters
        ----------
        params : dict
            Parameters of the statistical problem.
        """
        self.params = params
        self.n = params['n']
        self.p = params['p']
        self.theta = params['theta']
        if 'perturbation_class' not in params:
            self.perturbation_class = "JDS"
        else:
            self.perturbation_class = params['perturbation_class']
        self.minimax_bounds = []
        

    def get_true_theta(self):
        return self.theta

    @property
    def loss(self):
        if self.params['loss'] == "squared_error":
            return self.squared_loss
        else:
            raise NotImplementedError

    @abstractmethod
    def sample(self, num_copies):
        """
        num_copies : int
            Number of copies of the dataset to be sampled.
        """
        raise NotImplementedError

    @abstractmethod
    def get_estimators(self):
        raise NotImplementedError

    @abstractmethod
    def get_perturbations(self):
        raise NotImplementedError

    def squared_loss(self, theta_hat, theta):
        """
        theta_hat : np.array
            A 2D array (num_copies, p) containing the estimated parameter.
        theta : np.array
            A 1D array of shape (p,) containing the true parameter.
        """
        assert theta_hat.shape[1:] == theta.shape
        return np.sum(np.square(theta_hat - theta), axis=1)

    
    def _initialize_bounds(self, bounds=None):
        self.minimax_bounds = [bound for bound in bounds if bound.perturbation_type in self.hierarchy[self.perturbation_class]]
    
class GaussianLocation(StatisticalProblem):
    name="Gaussian Location"
    def __init__(self, **params):
        super().__init__(**params)
        self.sigma = params['sigma']
        assert(self.sigma.shape == (self.p,self.p))
        if len(self.sigma) == 1:
            self.trace = self.sigma
        else:
            self.trace = np.trace(self.sigma)
        self.ids_transition = np.sqrt(self.trace) / (self.n - 1)
        
        def IDS_bound(epsilon):
            if epsilon <= self.ids_transition:
                return 1/self.n * (epsilon + np.sqrt(self.trace))**2
            else:
                return epsilon ** 2 + self.trace/(self.n - 1)

        bounds = [
            Bound("CDS Rate", "CDS","C2", lambda epsilon: np.square(epsilon) + self.trace / self.n),
            Bound("IDS Rate", "IDS", "C3", IDS_bound),
            Bound("JDS Rate", "JDS", "C1", lambda epsilon: np.square(epsilon + np.sqrt(self.trace / self.n)))
        ]
        self._initialize_bounds(bounds)
        

    def sample(self, num_copies):
        """
        num_copies : int
            Number of copies of the dataset to be sampled.
        
        Returns
        -------
        np.array
            A 3D array of shape (num_copies, n, p) containing the data.
        """
        return np.random.multivariate_normal(mean=self.theta, cov=self.sigma, size=(num_copies,self.n))

    def get_estimators(self):
        return [MeanEstimator(), MedianEstimator()]
        
    def get_perturbations(self):
        zeta_mean_shift = np.sqrt(self.n/self.trace)
        perts = [
        perturbations.ConstantShiftFirst(),
        perturbations.ConstantShiftOnes(),
        perturbations.MeanAwayFromThetaShift(self.theta, zeta = zeta_mean_shift ),
        perturbations.GaussianIDSPert(self.theta, self.trace )
        ]
        valid_perturbations = [p for p in perts if p.perturbation_type in self.hierarchy[self.perturbation_class]]
        return valid_perturbations


class UniformLocation(StatisticalProblem):
    name = "Uniform Location"
    def __init__(self, **params):
        super().__init__(**params)
        assert(self.p ==1)

        n = self.n
        def JDS_bound(epsilon):
            cutoff = (np.sqrt(3/n) - 3 * np.sqrt(2/((n+1)*(n+2))))/(6*(np.sqrt(n)-1))
            if epsilon < cutoff:
                return (epsilon*np.sqrt(n) + 1/np.sqrt(2*(n+1)*(n+2)))**2
            else:
                return (epsilon + 1/np.sqrt(12*n))**2
        bounds = [
            Bound("CDS Rate", "CDS", "C2", lambda epsilon: epsilon**2 + 1/(2*(n+1)*(n+2))),
            Bound("IDS LB", "IDS", "C3",lambda epsilon: 0.614*np.power(epsilon,2/3)/n),
            #Bound("IDS LB Conj", "IDS", "C2", lambda epsilon: np.power(epsilon,4/3)),
            Bound("JDS UB", "JDS", "C1", JDS_bound)
        ]

        self._initialize_bounds(bounds)

    def sample(self, num_copies):
        """
        num_copies : int
            Number of copies of the dataset to be sampled.
        
        Returns
        -------
        np.array
            A 3D array of shape (num_copies, n, 1) containing the data.
        """
        return np.random.uniform(low=self.theta-1/2, high = self.theta+1/2, size=(num_copies,self.n,1))

    def get_estimators(self):
        estimators = [MeanEstimator(), MedianEstimator(), MaxMinEstimator()]
        for k in range(2, self.n//2+1):
            estimators.append(kthEstimator(int(k)))
        # for k in np.linspace(2, int(np.sqrt(self.n)),num=10):
        #     estimators.append(kthEstimator(int(k)))
        return estimators
        
    def get_perturbations(self):
        perts = [
        perturbations.ConstantShiftFirst(),
        perturbations.ConstantShiftOnes(),
        perturbations.ShiftMax(),
        ]
        for k in range(2, self.n//2+1):
            perts.append(perturbations.ShiftKth(int(k)))
        valid_perturbations = [p for p in perts if p.perturbation_type in self.hierarchy[self.perturbation_class]]
        return valid_perturbations
        
                
class LinearRegression(StatisticalProblem):
    name="Linear Regression"
    def __init__(self, **params):
        super().__init__(**params)
        self.sigma = params['sigma']
        assert(self.sigma.shape == (self.n,self.n))
        
        self.X = params['X']

        # For convenience
        X = self.X
        sigma = self.sigma
        self.is_diagonal = np.allclose(sigma,np.eye(sigma.shape[0])*sigma[0,0])
        sigma_inv = np.linalg.inv(sigma)
        P_X = X @ np.linalg.inv(X.T @ X) @ X.T
        P_X_Sigma = X @ np.linalg.inv(X.T @ sigma_inv @ X) @ X.T @ sigma_inv
        n = self.n
        
        if params['loss'] == "squared_error":
            lb_c1_bayes = 1/np.sqrt( np.trace(sigma @ P_X_Sigma))
            lb_c2_bayes = np.trace(np.linalg.inv(X.T @ np.linalg.inv(sigma) @ X))
            _,S,_ = np.linalg.svd(X)
            ub_c1 = 1/S[-1]
            ub_c2_helper = np.linalg.inv(X.T@X)
            ub_c2 = np.sqrt(np.trace(sigma @ X @ ub_c2_helper @ ub_c2_helper @ X.T))

            lb_c1_modulus = 1/(S[-1]**2)
            lb_c2_modulus = lb_c2_bayes

            bounds = [
                Bound( "LB Bayes", "JDS", "C2", lambda epsilon: (1 + epsilon * lb_c1_bayes) **2 * lb_c2_bayes ),
                Bound("LB Modulus","JDS", "C3", lambda epsilon: np.maximum(epsilon**2 * lb_c1_modulus,lb_c2_modulus)),
                Bound( "UB","JDS", "C1", lambda epsilon: (epsilon * ub_c1 + ub_c2) ** 2)
            ]
        elif params['loss'] == "prediction_error":
            
            rate_c = np.sqrt(np.trace(sigma @ P_X_Sigma)/n)
            bounds = [
                    Bound("Rate","JDS" ,"C1",  lambda epsilon: (epsilon/np.sqrt(self.n) + rate_c) ** 2)
                ]
        else:
            raise ValueError("Loss function not supported")

        self._initialize_bounds(bounds)
        self.P_X = P_X
        self.P_X_Sigma = P_X_Sigma
        u,s,vH = np.linalg.svd(X)
        self.s_min = s[-1]
        self.v = vH.T

    def sample(self, num_copies):
        """
        num_copies : int
            Number of copies of the dataset to be sampled.
        
        Returns
        -------
        np.array
            A 3D array of shape (num_copies, 1, n) containing the data.
        """
        noise = np.random.multivariate_normal(mean=np.zeros(self.n), cov=self.sigma, size=(num_copies))
        Y = self.X @ self.theta + noise
        return Y.reshape(num_copies, 1, self.n)

    def get_estimators(self):
        estimators = [LeastSquaresEstimator(self.X)]
        if not self.is_diagonal:
            estimators.append(GeneralizedLeastSquaresEstimator(self.X,self.sigma))
        return estimators
        
    def get_perturbations(self):
        zeta_pert = np.sqrt(1/np.trace(self.sigma @ self.P_X))
        zeta_gen_pert = np.sqrt(1/np.trace(self.sigma @ self.P_X_Sigma))
        zeta_constant_pert = 1#np.sqrt(self.n)
        perts = [
        perturbations.ConstantShiftFirst(),
        perturbations.ConstantShiftOnes(),
        perturbations.LinearRegressionPert(self.theta,self.X, zeta_pert, self.P_X),
        perturbations.GeneralizedLinearRegressionPert(self.theta,self.X, self.sigma, zeta_gen_pert, self.P_X_Sigma),
        perturbations.LinearRegressionSingularVecPert(self.theta,self.X, zeta_constant_pert, self.v)
        ]
        valid_perturbations = [p for p in perts if p.perturbation_type in self.hierarchy[self.perturbation_class]]
        return valid_perturbations
    
    @property
    def loss(self):
        if self.params['loss'] == "squared_error":
            return self.squared_loss
        elif self.params['loss'] == "prediction_error":
            return self.prediction_loss
        else:
            raise ValueError("Loss not recognized")
    
    def prediction_loss(self, theta_hat, theta):
        return np.linalg.norm(self.X @ (theta - theta_hat).T, axis = 0)**2 / self.n