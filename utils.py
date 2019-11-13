import numpy as np
from scipy.linalg import expm

def generate_random_rotation_matrix(d):
    M = 10 * (np.random.uniform(size=(d,d)))
    M = 0.5 * (M - M.T)
    R = expm(M)
    return R

def left_right_multiply_covariance(Sigma, W):
    # Sigma: (num_trials, num_electrodes, num_electrodes)
    # W: (num_components, num_electrodes)
    # Performs the operation: W * \Sigma * W^T, for 3-dimensional covariance input
    # Returns new covariance
    assert Sigma.ndim == 3
    assert W.ndim == 2
    assert Sigma.shape[1] == Sigma.shape[2]
    assert Sigma.shape[1] == W.shape[1]

    n_trials = Sigma.shape[0]
    n_components = W.shape[0]

    # Right multiply
    Sigma = np.dot(Sigma, W.T)

    # Left multiply
    Sigma = np.transpose(Sigma, (1,2,0))
    Sigma = np.dot(W, Sigma)
    Sigma = np.transpose(Sigma, (2,0,1))

    assert Sigma.shape == (n_trials, n_components, n_components)
    return Sigma

def left_right_multiply_covariance_2d(Sigma, W):
    # Sigma: (num_electrodes, num_electrodes)
    # W: (num_components, num_electrodes)
    # Computes W \Sigma W^T
    assert Sigma.shape[0] == Sigma.shape[1]
    assert Sigma.ndim == 2
    assert W.ndim == 2

    left = np.dot(W, Sigma)
    right = np.dot(left, W.T)
    return right

if __name__ == "__main__":
    n_tile = 5
    s = np.eye(2)
    s = np.expand_dims(s, 0)
    s = np.tile(s, (n_tile,1,1))
    w = np.eye(2)*2
    s = left_right_multiply_covariance(s, w)
    for i in range(n_tile):
        assert np.array_equal(s[i], np.array([[4,0],[0,4]]))
    print "Test case 1 pass"

    s = np.eye(2)
    w = np.eye(2)*2
    s = left_right_multiply_covariance_2d(s, w)
    assert np.array_equal(s, np.array([[4,0],[0,4]]))
    print "Test case 2 pass"

    s = np.eye(2)
    w = np.eye(2)
    w = np.tile(w, (1,2))
    w = w.T
    s = left_right_multiply_covariance_2d(s, w)
    assert np.array_equal(s, np.tile(np.eye(2), (2,2)))
    print "Test case 3 pass"
    

