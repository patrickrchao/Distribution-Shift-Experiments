import numpy as np

class Perturbation:
    perturbation_type = None
    tol = 1e-3

    def __init__(self):
        pass
    
    def __call__(self, epsilon, data, **kwargs):
        pass

    def check_perturbation(self, epsilon, data, **kwargs):
        """
        Parameters
        ----------
        epsilon : float
            The perturbation parameter.
        data : np.array 
            A 3D array of shape (num_copies, n, p) containing the data

        Returns
        -------
        boolean whether the perturbation is valid or not
        """
        assert(len(data.shape) == 3)
        perturbed_data = self(epsilon, data, *kwargs)
        empirical_distance = np.sqrt(np.mean(np.sum(np.square(data - perturbed_data), axis=(2))))
        epsilon_sq = epsilon ** 2

        if (empirical_distance - epsilon)/epsilon > self.tol:
            print(f"Perturbation is invalid. Empirical distance is {empirical_distance:>.4f} and epsilon is {epsilon:>.4f}.")
            return False
        elif empirical_distance/epsilon_sq < 0.95:
            print(f"WARNING: Empirical distance is {empirical_distance:>.4f} and epsilon is {epsilon:>.4f}.")
        else:
            print(f"Perturbation is valid. Empirical distance is {empirical_distance:>.4f} and epsilon  is {epsilon:>.4f}.")
        return True



class ConstantShiftFirst(Perturbation):
    perturbation_type = "CDS"

    def __call__(self, epsilon, data):
        """
        Perturb the data by adding a constant to the first feature.

        Parameters
        ----------
        epsilon : float
            The perturbation parameter.
        data : np.array
            A 3D array of shape (num_copies, n, p) containing the data.
        """
        pert = data.copy()
        pert[:,:,0] += epsilon
        return pert

class ConstantShiftOnes(Perturbation):
    perturbation_type = "CDS"

    def __call__(self, epsilon, data):
        """
        Perturb the data by adding a constant to all features

        Parameters
        ----------
        epsilon : float
            The perturbation parameter.
        data : np.array
            A 3D array of shape (num_copies, n, p) containing the data.
        """
        p = data.shape[2]
        return data + epsilon/np.sqrt(p)

class MeanAwayFromThetaShift(Perturbation):
    def __init__(self, theta, zeta):
        super().__init__()
        self.theta = theta
        self.zeta = zeta
        
    perturbation_type = "JDS"

    def __call__(self, epsilon, data):
        """
        Perturb the data by pushing $X_i$ from $\bar X-\theta$

        Parameters
        ----------
        epsilon : float
            The perturbation parameter.
        data : np.array
            A 3D array of shape (num_copies, n, p) containing the data.
        """
        sample_mean = np.mean(data, axis=1)
        shift = self.zeta * (sample_mean - self.theta)
        # Add dimension to shift
        shift = shift.reshape(shift.shape[0], 1, shift.shape[1])
        return data + epsilon * shift

class GaussianIDSPert(Perturbation):
    def __init__(self, theta, trace):
        super().__init__()
        self.theta = theta
        self.trace = trace
        
    perturbation_type = "IDS"

    def __call__(self, epsilon, data):
        """
        Perturb the data by using the IDS perturbation

        Parameters
        ----------
        epsilon : float
            The perturbation parameter.
        data : np.array
            A 3D array of shape (num_copies, n, p) containing the data.
        """
        p = data.shape[2]
        n = data.shape[1]
        delta = np.ones(p) / np.sqrt(p)
        zeta, psi = self.get_zeta_psi(epsilon, n)
        shift = zeta * (data - self.theta) + psi * delta

        # Add dimension to shift
        return data + shift
    
    def get_zeta_psi(self, epsilon, n):
        """
        Returns the zeta and psi parameters for the IDS perturbation
        """
        zeta = np.minimum(epsilon/np.sqrt(self.trace), 1/(n-1))
        psi = np.sqrt(np.maximum(0, epsilon**2 - self.trace / ((n-1)**2)))
        return zeta, psi

class ShiftMax(Perturbation):
    perturbation_type = "JDS"
    def __call__(self, epsilon, data):
        """
        Perturb the data by using the IDS perturbation

        Parameters
        ----------
        epsilon : float
            The perturbation parameter.
        data : np.array
            A 3D array of shape (num_copies, n, 1) containing the data.
        """
        assert(data.shape[2] == 1)
        n = data.shape[1]
        return data + epsilon * np.sqrt(n)* (data >= np.max(data,axis=1).reshape(-1,1,1))
    
class ShiftKth(Perturbation):
    perturbation_type = "JDS"
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def __call__(self, epsilon, data):
        """
        Perturb the data by using the IDS perturbation

        Parameters
        ----------
        epsilon : float
            The perturbation parameter.
        data : np.array
            A 3D array of shape (num_copies, n, 1) containing the data.
        """
        assert(data.shape[2] == 1)
        n = data.shape[1]
        kth_smallest = np.sort(data,axis=1)[:,self.k-1,:].reshape(-1,1,1)
        return data + epsilon * np.sqrt(n/self.k) * (data <= kth_smallest)

class LinearRegressionPert(Perturbation):
    perturbation_type = "JDS"
    def __init__(self, theta, X, zeta, P_X):
        super().__init__()
        self.theta = theta
        self.X = X
        self.zeta = zeta
        # Pass in P_X since it is already precomputed
        self.P_X = P_X
        
    def __call__(self, epsilon, Y):
        """
        Perturb the data by using the JDS perturbation

        Parameters
        ----------
        epsilon : float
            The perturbation parameter.
        data : np.array
            A 3D array of shape (num_copies, n, 1) containing the data.
        """
        
        shift = self.P_X @ (Y.squeeze().T) - (self.X @ self.theta).reshape(-1,1)
        
        return Y + epsilon * self.zeta * shift.T[:,None,:]

class GeneralizedLinearRegressionPert(Perturbation):
    perturbation_type = "JDS"
    def __init__(self, theta, X, sigma, zeta, P_X_Sigma):
        super().__init__()
        self.theta = theta
        self.X = X
        self.zeta = zeta
        self.sigma = sigma
        # Pass in P_X_Sigma since it is already precomputed
        self.P_X_Sigma = P_X_Sigma
        
    def __call__(self, epsilon, Y):
        """
        Perturb the data by using the JDS perturbation

        Parameters
        ----------
        epsilon : float
            The perturbation parameter.
        data : np.array
            A 3D array of shape (num_copies, 1, n) containing the data.
        """
        
        shift = self.P_X_Sigma @ (Y.squeeze().T) - (self.X @ self.theta).reshape(-1,1)
        return Y + epsilon * self.zeta * shift.T[:,None,:]

class LinearRegressionSingularVecPert(Perturbation):
    perturbation_type = "JDS"
    def __init__(self, theta, X, zeta, v):
        super().__init__()
        self.theta = theta
        self.X = X
        self.zeta = zeta
        self.v = v
       
        self.shift_dir = X @ v[:,-1]
        #print(np.linalg.norm(self.shift_dir))
        self.shift_dir = self.shift_dir / np.linalg.norm(self.shift_dir)
        
        #print("1",X.shape,self.shift_dir.shape,v.shape)
        
    def __call__(self, epsilon, Y):
        """
        Perturb the data by using the IDS perturbation

        Parameters
        ----------
        epsilon : float
            The perturbation parameter.
        data : np.array
            A 3D array of shape (num_copies, n, 1) containing the data.
        """
        
        #shift = self.P_X @ (Y.squeeze().T) - (self.X @ self.theta).reshape(-1,1)
        shift = self.shift_dir.reshape(1,1,-1)
        #print("2",Y.shape,shift.shape)
        
        #print(np.linalg.norm(self.zeta * shift,axis=1).flatten())
        #return Y + epsilon * self.zeta * shift#shift.T[:,:,None]
        # pert_y = Y + epsilon * self.zeta* shift
        # pred_1 = np.linalg.inv(self.X.T@self.X)@ (self.X.T)@ Y
        # pred_2 = np.linalg.inv(self.X.T@self.X)@ (self.X.T)@ pert_y
        # print(f"{np.linalg.norm(pred_1-pred_2,axis=1)} should be close to np.sqrt(n)ep/s_min={epsilon/0.1*self.zeta}")
        return Y + epsilon * self.zeta* shift#shift.T[:,:,None]
    
            
        