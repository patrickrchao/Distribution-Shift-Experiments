import numpy as np
import matplotlib.pyplot as plt
import perturbations
import estimators
from statistical_problems import GaussianLocation, UniformLocation, LinearRegression
from simulation import Simulation
#################################
##### SIMULATION PARAMETERS #####
#################################
alpha_values = np.linspace(-2.5, 0, 40)
num_copies = 10000
#################################

# Define problem parameters
gaussian_params = {
    'n': 10,
    'p': 3,
    'theta': np.array([0.0, 0.0, 0.0]),
    'sigma': np.diag([1/6, 2/6, 3/6]),
    
    'perturbation_class' : 'JDS'
}

uniform_params = {
    'n':50,
    'p':1,
    'theta' : np.array([3]),
    'perturbation_class' : 'JDS'
}

linear_regression_params = {
    'n':10,
    'p':5,
    'theta' : np.ones(5),
    'perturbation_class' : 'JDS'
}


def get_statistical_problem(problem_type, loss_type = "squared_error", sigma = None):
    new_params = {"loss": loss_type}
    if sigma is not None:
        new_params["sigma"] = sigma
    if problem_type == "Gaussian":
        # Create a GaussianMean problem instance
        problem = GaussianLocation(**(gaussian_params | new_params))
    elif problem_type == "Uniform":
        # Create a UniformMean problem instance
        problem = UniformLocation(**(uniform_params | new_params))
    elif problem_type == "Linear Regression":
        # Create a Linear Regression problem instance
        n,p = linear_regression_params['n'],linear_regression_params['p']
        X = np.random.normal(size = (n,p))/np.sqrt(n)
        linear_regression_params['X'] = X
        problem = LinearRegression(**(linear_regression_params | new_params))
    else:
        raise NotImplementedError
    return problem

# Define epsilon values 

display_est = "best"

for problem_type in ["Gaussian","Linear Regression","Uniform"]:
    all_sigma = [None]
    losses = ["squared_error"]
    if problem_type == "Linear Regression":
        all_sigma = [np.eye(10)/100, np.diag([(i+1)/200 for i in range(10)])]
        losses.append("prediction_error")
    for loss in losses:
        for i, sigma in enumerate(all_sigma):
            print(f"Running {problem_type} with {loss} loss and sigma num {i+1}/{len(all_sigma)}")
            problem = get_statistical_problem(problem_type, sigma=sigma, loss_type=loss)
            epsilon_values = np.power(problem.n, alpha_values)

            # Create and run the simulation
            
            simulation = Simulation(problem, epsilon_values, num_copies, display_est)
            results = simulation.run_simulation()
            
            plt.rcParams['figure.figsize'] = [8, 6]

            simulation.display_est = display_est
            simulation.generate_plot(results,show=False)