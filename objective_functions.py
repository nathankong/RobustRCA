import numpy as np
from utils import left_right_multiply_covariance_2d

def symmetric_kl_objective(R, Sigma1, Sigma2, d):
    # We want to maximize this objective function. This objective function deals 
    # with *average* class distributions
    # Parameters:
    #     R: rotation matrix (num_features, num_features)
    #     P: whitening matrix (num_features, num_features)
    #     Sigma1, Sigma2: No trial information since covariances have been
    #                     averaged already. (1, num_features, num_features)

    # TODO: Deal with code optimization later (i.e. truncation part)

    assert Sigma1.shape == Sigma2.shape
    #assert P.shape == Sigma1[0].shape
    assert R.shape[0] == R.shape[1] # Check if square
    assert np.allclose(np.dot(R, R.T), np.eye(R.shape[0])) # Check orthogonality
    Sigma1 = Sigma1[0]
    Sigma2 = Sigma2[0]

    D = Sigma1.shape[0]

    Id = np.eye(D, D)
    Id = Id[:d, :] # Truncated identity
    V = R.T
    V_trunc = np.dot(Id, V) # Truncate rotation matrix

    Sigma1_term = left_right_multiply_covariance_2d(Sigma1, V_trunc)
    Sigma2_term = left_right_multiply_covariance_2d(Sigma2, V_trunc)

    reg = np.eye(Sigma1_term.shape[0]) * 1e-8
    inv_Sigma1 = np.linalg.inv(Sigma1_term + reg)
    inv_Sigma2 = np.linalg.inv(Sigma2_term + reg)

    obj = 0.5 * np.trace(np.dot(inv_Sigma1, Sigma2_term) + np.dot(inv_Sigma2, Sigma1_term)) - d
    return obj

def grad_symmetric_kl_obj(R, Sigma1, Sigma2, d):
    # Computes gradient matrix for symmetric KL objective function.
    # Parameters:
    #     R: rotation matrix (num_features, num_features)
    #     P: whitening matrix (num_features, num_features)
    #     Sigma1, Sigma2: No trial information since covariances have been
    #                     averaged already. (1, num_features, num_features)

    assert Sigma1.shape == Sigma2.shape
    #assert P.shape == Sigma1[0].shape
    assert R.shape[0] == R.shape[1]
    assert np.allclose(np.dot(R, R.T), np.eye(R.shape[0])) # Check orthogonality
    Sigma1 = Sigma1[0]
    Sigma2 = Sigma2[0]

    D = Sigma1.shape[0]
    Id = np.eye(D, D)
    Id = Id[:d, :] # Truncated identity
    V = R.T
    V_trunc = np.dot(Id, V) # Truncate rotation matrix

    Sigma1_bar = left_right_multiply_covariance_2d(Sigma1, V_trunc)
    Sigma2_bar = left_right_multiply_covariance_2d(Sigma2, V_trunc)
    Sigma1_tilde = Sigma1
    Sigma2_tilde = Sigma2
    inv_Sigma1_bar = np.linalg.inv(Sigma1_bar)
    inv_Sigma2_bar = np.linalg.inv(Sigma2_bar)

    # Gradient computed using CSP term from Table II of Samek et al. 2014 review
    term1 = np.dot(np.dot(inv_Sigma2_bar, Id), Sigma2_tilde)
    term2 = np.dot(np.dot(np.dot(inv_Sigma1_bar, Sigma2_bar), inv_Sigma1_bar), np.dot(Id, Sigma1_tilde))
    term3 = np.dot(np.dot(inv_Sigma1_bar, Id), Sigma1_tilde)
    term4 = np.dot(np.dot(np.dot(inv_Sigma2_bar, Sigma1_bar), inv_Sigma2_bar), np.dot(Id, Sigma2_tilde))

    grad = np.dot(np.dot(Id.T, term1-term2+term3-term4), R)
    grad = grad - grad.T # Top left and bottom right blocks should be zeros
    return grad

def sum_symmetric_kl_obj(R, Sigma1, Sigma2, V):
    # Loss function deals with *sum of trial-wise* covariances
    pass

def grad_sum_symmetric_kl_obj():
    # Gradient matrix for sum symmetric KL obj
    # We want to maximize this objective function.
    pass


