import unittest
import numpy as np
from statistical_problems import GaussianMean, UniformLocation
from estimators import MeanEstimator
from perturbations import ConstantShiftFirst, ConstantShiftOnes, GaussianIDSPert
from simulation import Simulation


class TestGaussianMean(unittest.TestCase):
    def setUp(self):
        self.problem = GaussianMean(n=100, p=3, theta=np.array(
            [0, 1, 2]), sigma=np.diag([2,3,4]), loss="squared_error")

    def test_sample(self):
        num_copies = 5
        data = self.problem.sample(num_copies)
        self.assertEqual(
            data.shape, (num_copies, self.problem.n, self.problem.p))

    def test_squared_loss(self):
        theta = np.array([0, 1, 2])
        theta_hat = np.array([[0, 1, 2], [1, 2, 3], [0, 0, 0], [1, 1, 1]])
        expected_losses = np.array([0, 3, 5, 2])
        computed_losses = self.problem.squared_loss(theta_hat, theta)
        np.testing.assert_almost_equal(
            computed_losses, expected_losses, decimal=8)


class TestMeanEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = MeanEstimator()

    def test_estimate(self):
        data = np.random.normal(size=(5, 100, 3))
        mean_estimates = self.estimator(data)
        self.assertEqual(mean_estimates.shape, (5, 3))


class TestPerturbations(unittest.TestCase):
    def test_constant_shift_first(self):
        perturbation = ConstantShiftFirst()
        data = np.random.normal(size=(5, 100, 3))
        epsilon = 0.5
        perturbed_data = perturbation(epsilon, data)
        temp = data.copy()
        temp[:, :, 0] += epsilon
        np.testing.assert_almost_equal(
            perturbed_data[:, :, :], temp, decimal=8)
        self.assertTrue(perturbation.check_perturbation(epsilon, data))

    def test_constant_shift_ones(self):
        perturbation = ConstantShiftOnes()
        data = np.random.normal(size=(5, 100, 3))
        epsilon = 0.5
        perturbed_data = perturbation(epsilon, data)
        np.testing.assert_almost_equal(
            perturbed_data, data + epsilon/np.sqrt(3), decimal=8)

        self.assertTrue(perturbation.check_perturbation(epsilon, data))
    
    def test_gaussian_pert(self):
        params={"n":5,
        "p":3,
        "theta":np.array(
            [0, 1, 2]),
        "sigma":np.diag([1,2,3]),
        "loss":"squared_error"
        }
        problem = GaussianMean(**params)
        epsilon_values = np.linspace(0,1,10)
        num_copies = 10000
        self.simulation = Simulation(problem, epsilon_values, num_copies)
        data = np.random.multivariate_normal(params["theta"],params["sigma"],size=(num_copies,params["n"]))
        trace = np.trace(params["sigma"])
        n = params["n"]
        perturbation = GaussianIDSPert(params["theta"],trace)
        for ep in epsilon_values:
            zeta, psi = perturbation.get_zeta_psi(ep, n)
            
            np.testing.assert_almost_equal(psi, np.sqrt(np.maximum(0,ep**2-trace/((n-1)**2))), decimal=8)
            np.testing.assert_almost_equal(zeta, np.minimum(np.sqrt(ep**2/trace),1/(n-1)), decimal=8)
            
            self.assertTrue(perturbation.check_perturbation(ep, data))
    
    def test_uniform_pert(self):
        params={"n":5,
        "p":1,
        "theta":np.array(
            [5]),
        "loss":"squared_error"
        }
        problem = UniformLocation(**params)
        epsilon_values = np.linspace(0,1,10)
        num_copies = 10000
        self.simulation = Simulation(problem, epsilon_values, num_copies)
        n = params["n"]
        theta = params['theta']
        data = np.random.uniform(low=theta-1/2, high = theta+1/2, size=(num_copies,n,1))

        
        perturbations = problem.get_perturbations()
        for ep in epsilon_values:
            for perturbation in perturbations:
                perturbed_data = perturbation(ep, data)
                self.assertTrue(perturbation.check_perturbation(ep, data))

class TestSimulation(unittest.TestCase):
    def setUp(self):
        problem = GaussianMean(n=100, p=3, theta=np.array(
            [0, 1, 2]), sigma=np.diag([1,2,3]), loss="squared_error")
        epsilon_values = [0.1, 0.2, 0.3]
        num_copies = 5
        self.simulation = Simulation(problem, epsilon_values, num_copies)

    # def test_run_simulation(self):
    #     results = self.simulation.run_simulation()
    #     for estimator in self.simulation.estimators:
    #         self.assertIn(estimator.name, results)
    #         for epsilon in self.simulation.epsilon_values:
    #             self.assertIsInstance(results[estimator][epsilon], float)


if __name__ == '__main__':
    unittest.main()
