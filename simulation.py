import numpy as np
import matplotlib.pyplot as plt 
# Increase plot size and figure size
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 100
# Increase plot text size
# font = {
#         #'weight' : 'bold',
#         'size'   : 22}
# plt.rc('font', **font)
plt.rcParams.update({'font.size': 22})
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
from tqdm import tqdm
class Simulation:
    def __init__(self, problem, epsilon_values, num_copies, display_est="best"):
        self.problem = problem
        self.epsilon_values = epsilon_values
        self.num_copies = num_copies
        self.estimators = problem.get_estimators()
        self.perturbations = problem.get_perturbations()
        self.log_epsilon_values = self.log_base_n(epsilon_values)
        assert(display_est in ["all","best"])
        self.display_est = display_est
        self.include_title = False

    def run_simulation(self):
        results = {}
        for estimator in self.estimators:
            risk_per_epsilon = np.zeros(len(self.epsilon_values))
            results[estimator.name] = risk_per_epsilon
            for i,epsilon in enumerate(tqdm(self.epsilon_values)):
                max_risk = -np.inf
                for perturbation in self.perturbations:
                    
                    # Generate perturbed data
                    data = self.problem.sample(self.num_copies)
                    perturbed_data = perturbation(epsilon,data)
                    #perturbation.check_perturbation(epsilon,data)
                    # Compute the estimator on the perturbed data
                    theta_hat = estimator(perturbed_data)

                    # Compute the risk
                    risks = self.problem.loss(theta_hat, self.problem.get_true_theta())

                    # Update the maximum risk for the current estimator and epsilon value
                    max_risk = max(max_risk, np.mean(risks))

                risk_per_epsilon[i] = max_risk
        return results

    def generate_plot(self, results,show=True):
        plt.figure()
        # Plot Minimax Bounds
        
        all_bound_values = {}
        for bound in self.problem.minimax_bounds:
            bound_values = [bound(epsilon) for epsilon in self.epsilon_values]
            log_bounds = self.log_base_n(bound_values)

            all_bound_values[bound.name] = log_bounds
            plt.plot(self.log_epsilon_values, log_bounds, label=bound.name, lw=5, color = bound.color)

        if self.problem.name == "Uniform Location":
            lb_bounds = np.maximum(all_bound_values["IDS LB"],all_bound_values["CDS Rate"])
            ub_bounds = all_bound_values["JDS UB"]
            #shaded_region_y = lb_bounds + ub_bounds[::-1]
            plt.fill_between(self.log_epsilon_values,y1=lb_bounds,y2=ub_bounds,hatch="\\",alpha=0.2,color="tomato",edgecolor="black")

        # Plot estimators
        if self.display_est == "all":
            for i,(name, estimator_results) in enumerate(results.items()):
                #plt.scatter(self.log_epsilon_values, self.log_base_n(estimator_results), label=name, marker='x')
                plt.scatter(self.log_epsilon_values, self.log_base_n(estimator_results), label=name, marker='x',color=f"C{i+3}",zorder=5)
        else:
            best_results = np.ones(len(self.epsilon_values)) * np.inf
            for name, estimator_results in results.items():
                best_results = np.minimum(best_results, estimator_results)
            plt.scatter(self.log_epsilon_values, self.log_base_n(best_results), label="Empirical", marker="x",color="C0",zorder=5, s =120)
    

        # Plot vertical dashed line for Gaussian location
        if self.problem.name == "Gaussian Location":
            
            plt.axvline(x= self.log_base_n(self.problem.ids_transition),label=r"$\varepsilon=\sqrt{p}\sigma/(n-1)$", color="slategray",linestyle="dashed",lw=3)
        plt.xlabel(r'Power $\alpha$   ($\varepsilon=n^\alpha$)')
        plt.ylabel(r'Log Risk / Log n')
        plt.legend()
        if self.problem.name == "Linear Regression":
            plt_title = f'Empirical LR {self.problem.params["loss"].replace("_"," ").title()} Log Minimax Risk'
        else:
            plt_title = f'Empirical {self.problem.perturbation_class} {self.problem.name} Log Minimax Risk'
        if self.include_title:
            plt.title(plt_title)
        plt.grid(axis="x")
        file_name = f'./Images/{self.problem.name.replace(" ","_")}_{self.problem.perturbation_class}_n_{self.problem.n}'
        if self.problem.name == "Linear Regression":
            file_name += f'_{self.problem.params["loss"]}'
            sigma = self.problem.sigma
            if not self.problem.is_diagonal:
                file_name += "_noniso"
        file_name += ".pdf"
        plt.savefig(file_name,bbox_inches='tight')
        if show:
            plt.show()
        

    def log_base_n(self, value):
        return np.log(value) / np.log(self.problem.n)