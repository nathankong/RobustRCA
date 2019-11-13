import numpy as np
from CovarianceEstimators import MaximumLikelihoodCovarianceEstimator
from DivergenceBasedSpatialFilter import KLDivergenceSpatialFilter
from objective_functions import symmetric_kl_objective, grad_symmetric_kl_obj

def main_all():
    fname = "/home/groups/amnorcia/eeg_data/kaneshiro_data/kaneshiro_eeg.npy"
    d = np.load(fname)
    d = np.transpose(d, (0,1,2,4,3))
    print d.shape

    c = MaximumLikelihoodCovarianceEstimator("mle")
    covXX, covXY, _ = compute_rca_covariance_subj_condition(d, c)

    # TODO: Complete function

def main_precomputed_covariance():
    covXX = np.load("covariance_matrices/avgCovXX.npy") + np.load("covariance_matrices/avgCovYY.npy")
    covXY = np.load("covariance_matrices/avgCovXY.npy")
    covXX = np.expand_dims(covXX, 0)
    covXY = np.expand_dims(covXY, 0)

    obj_func = symmetric_kl_objective
    grad_func = grad_symmetric_kl_obj
    tol = 1e-3
    n_iter = 20
    ls_niter = 20
    ksf = KLDivergenceSpatialFilter("kl_div", obj_func, grad_func, tol, n_iter, ls_niter, verbose=True)
    V_star = ksf.compute_spatial_filter(covXY, covXX, d=3)
    print V_star.shape


if __name__ == "__main__":
    np.random.seed(0)
    main_precomputed_covariance()

