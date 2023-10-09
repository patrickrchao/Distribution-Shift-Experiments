#  &nbsp; **Statistical Estimation Under Distribution Shift: Wasserstein Perturbations and Minimax Theory** &nbsp; 


[![arXiv](https://img.shields.io/badge/statML-arXiv%3A2308.01853-b31b1b)](https://arxiv.org/abs/2308.01853)

</div>

<!-- DESCRIPTION -->
## Abstract
Distribution shifts are a serious concern in modern statistical learning as they can systematically change the properties of the data away from the truth.
    We focus on Wasserstein distribution shifts, where every data point may undergo a slight perturbation, as opposed to the Huber contamination model where a fraction of observations are outliers. 
    We consider perturbations that are either independent 
    or coordinated joint shifts across data points.
    We analyze several important statistical problems, including location estimation, linear regression, and non-parametric density estimation. 
    Under a squared loss for mean estimation and prediction error in linear regression, we find the \emph{exact minimax risk}, a least favorable perturbation, and show that the sample mean and least squares estimators are respectively optimal. For other problems, we provide nearly optimal estimators and precise finite-sample bounds.
    We also introduce several tools for bounding the minimax risk under distribution shift, such as a smoothing technique for location families, and generalizations of classical tools including least favorable sequences of priors, the modulus of continuity, 
    as well as Le Cam's, Fano's, and Assouad's methods.


## Reproducing Experiments
### Run All Simulations
To reproduce the simulations in the paper, run the following:
```
python3 run_all_experiments.py
```
This runs the Gaussian and Uniform problems, as well as the linear regression problem under homoskedastic and heteroskedastic error and both squared and prediction error.
The uniform experiment takes the bulk of computational runtime due to computing many estimators and perturbations.

The outputs are saved by default in the `Images` folder.


### Run Individual Simulation
It is also possible to run individual simulations with custom parameters. In `run_experiments.ipynb', one may choose the statistical problem, parameters loss function, and number of simulated copies.



## Modifying Code
The codebase may be modified to different statistical problems and estimators. These may be added in `statistical_problems.py` and `estimators.py`.



## Citation
For any questions, feel free to contact us at `pchao@wharton.upenn.edu`.

```
@misc{chao2023statistical,
      title={Statistical Estimation Under Distribution Shift: Wasserstein Perturbations and Minimax Theory}, 
      author={Patrick Chao and Edgar Dobriban},
      year={2023},
      eprint={2308.01853},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```