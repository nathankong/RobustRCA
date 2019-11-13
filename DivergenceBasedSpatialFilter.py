import numpy as np
from scipy.linalg import fractional_matrix_power, expm

from objective_functions import symmetric_kl_objective
from utils import generate_random_rotation_matrix, \
    left_right_multiply_covariance

class DivergenceFrameworkSpatialFiltering(object):
    def __init__(self, divergence_type, tolerance):
        self.divergence_type = divergence_type
        self.tolerance = tolerance

    def compute_spatial_filter(self, Sigma1, Sigma2, d=None):
        raise NotImplementedError

class KLDivergenceSpatialFilter(DivergenceFrameworkSpatialFiltering):
    def __init__(self, divergence_type, obj_func, gradient_func, tolerance, n_iter, line_search_n_iter, verbose=False):
        super(KLDivergenceSpatialFilter, self).__init__(divergence_type, tolerance)

        self.objective_func = obj_func
        self.gradient_func = gradient_func
        self.n_iter = n_iter
        self.line_search_n_iter = line_search_n_iter
        self.verbose = verbose

    def compute_spatial_filter(self, tSigma1, tSigma2, d=None):
        # tSigma1: (num_epochs, num_features, num_features)
        # tSigma2: (num_epochs, num_features, num_features)
        assert tSigma1.shape == tSigma2.shape
        assert tSigma1.shape[1] == tSigma1.shape[2]
        n_features = tSigma1.shape[1]
        tSigma1_orig = tSigma1
        tSigma2_orig = tSigma2

        if d is None:
            d = 3

        # Average covariance across trials before computing whitening matrix
        Sigma1 = np.mean(tSigma1, axis=0)
        Sigma2 = np.mean(tSigma2, axis=0)
        Sigma1 += np.eye(Sigma1.shape[0])*1e-10
        Sigma2 += np.eye(Sigma2.shape[0])*1e-10
        P = fractional_matrix_power(Sigma1 + Sigma2, -0.5) # TODO: P should be symmetric. check.
        R = generate_random_rotation_matrix(n_features)

        print P
        assert 0

        # Whiten the covariances
        wSigma1 = left_right_multiply_covariance(tSigma1, P)
        wSigma2 = left_right_multiply_covariance(tSigma2, P)
        tSigma1 = left_right_multiply_covariance(wSigma1, R)
        tSigma2 = left_right_multiply_covariance(wSigma2, R)

        prev_obj_value = np.finfo(np.float).min
        for i in range(self.n_iter):
            if self.verbose:
                print "Iteration {}/{}; Objective: {}".format(i+1, self.n_iter, prev_obj_value)

            obj_value = self.objective_func(R, wSigma1, wSigma2, d)

            print obj_value

            gradient = self.gradient_func(R, tSigma1, tSigma2, d)
            t = self._compute_optimal_step_size(gradient, obj_value, tSigma1, tSigma2, d)
            U = expm(t * gradient)

            R = np.dot(U, R)
            tSigma1 = left_right_multiply_covariance(tSigma1, U)
            tSigma2 = left_right_multiply_covariance(tSigma2, U)

            # TODO: I don't think you need the abs, because the objective
            # function should keep increasing each iteration.
            if np.abs(obj_value - prev_obj_value) < self.tolerance:
                print "Converged with objective value: {}".format(obj_value)
                break

            prev_obj_value = obj_value

        # Take first `d' eigenvectors
        Sigma1 = np.mean(tSigma1, axis=0) # TODO: not sure if correct to average across trials here
        V = np.dot(R, P)[:d,:].T

        W, G = np.linalg.eigh(np.dot(V.T, Sigma1), V)
        W, G = self._sort_eig_descending(W, G)
        assert G.shape == (d,d)

        V_star = np.dot(V, G)
        return V_star

    def _compute_optimal_step_size(self, grad_matrix, current_obj_value, tSigma1, tSigma2, d):
        # Returns scalar from line search

        # grad_matrix should be skew symmetric
        assert np.allclose(grad_matrix + grad_matrix.T, np.zeros(grad_matrix.shape))
        search_direction = grad_matrix / (0.5*np.linalg.norm(grad_matrix, "fro"))
        t = 1.
        alpha = 0.9
        for i in range(self.line_search_n_iter):
            R = expm(t*grad_matrix)

            obj_value = self.objective_func(R, tSigma1, tSigma2, d)
            print "  Line search iteration {}/{}; Old: {}; New: {}".format(i+1, self.line_search_n_iter, current_obj_value, obj_value)
            if obj_value > current_obj_value:
                print "  ==Line search succeeded."
                return t

            t = alpha * t
        print "  ==Line search failed."
        return t

    def _sort_eig_descending(self, W, G):
        # W: eigenvalues, G: eigenvectors
        assert W.size == G.shape[1]
        idx = W.argsort()[::-1]
        W = W[idx]
        G = G[:,idx]
        return W, G

