import numpy as np
from itertools import combinations
from CovarianceEstimators import CovarianceEstimatorBase

def compute_rca_covariance_subj_condition(data, covariance_estimator):
    # Computes the within-trial and across-trial covariance matrices for a data set
    #
    # Parameters:
    #     data: n_subjects, n_conditions x n_trial x n_samples (i.e. n_time) x n_channels
    #     covariance_estimator: instance of CovarianceEstimatorBase (contains function to 
    #                           compute covariance
    # Returns:
    #     covXX: within trial covariance
    #     covYY: within trial covariance
    #     covXY: across trial covariance

    assert isinstance(covariance_estimator, CovarianceEstimatorBase)

    n_subjects = data.shape[0]
    n_conditions = data.shape[1]
    n_trials = data.shape[2]
    n_samples = data.shape[3]
    n_channels = data.shape[4]

    allCovXX = np.zeros((n_subjects, n_conditions, n_channels, n_channels))
    allCovXY = np.zeros((n_subjects, n_conditions, n_channels, n_channels))
    allCovYY = np.zeros((n_subjects, n_conditions, n_channels, n_channels))
    for i in range(n_subjects):
        print "Subject {}/{}".format(i+1, n_subjects)
        for j in range(n_conditions):
            print "  Condition {}/{}".format(j+1, n_conditions)
            d = data[i,j,:,:,:]
            covXX, covXY, covYY = _compute_rca_covariance(d, covariance_estimator)
            allCovXX[i,j,:,:] = covXX
            allCovXY[i,j,:,:] = covXY
            allCovYY[i,j,:,:] = covYY

    avgCovXX = np.mean(allCovXX, axis=(0,1))
    avgCovXY = np.mean(allCovXY, axis=(0,1))
    avgCovYY = np.mean(allCovYY, axis=(0,1))

    assert np.allclose(avgCovXX, avgCovYY)
    return avgCovXX, avgCovXY, avgCovYY

def _compute_rca_covariance(data, covariance_estimator):
    # Computes the within-trial and across-trial covariance matrices for a specific
    # subject and condition data.
    #
    # Parameters:
    #     data (single subject data, single condition): n_trial x n_samples (i.e. n_time) 
    #                                                   x n_channels
    #     covariance_estimator: instance of CovarianceEstimatorBase (contains function to 
    #                           compute covariance
    # Returns:
    #     covXX: within trial covariance (n_channels, n_channels)
    #     covYY: within trial covariance (n_channels, n_channels)
    #     covXY: across trial covariance (n_channels, n_channels)

    n_trials = data.shape[0]
    n_samples = data.shape[1]
    n_channels = data.shape[2]
    pair_idx = get_trial_pairs(n_trials)

    # Change shape to: (n_channels, n_samples, n_pairs)
    X = np.transpose(data[pair_idx[:,0],:,:], (2,0,1))
    Y = np.transpose(data[pair_idx[:,1],:,:], (2,0,1))

    # Shape: (n_channels, n_samples*n_pairs)
    X = np.reshape(X, (n_channels, -1))
    Y = np.reshape(Y, (n_channels, -1))

    covXX = covariance_estimator.compute_covariance(X, X)
    covXY = covariance_estimator.compute_covariance(X, Y)
    covYY = covariance_estimator.compute_covariance(Y, Y)

    assert np.allclose(covXX, covYY)
    assert covXX.shape == covXY.shape == covYY.shape
    assert covXX.shape == (n_channels, n_channels)

    return covXX, covXY, covYY

def get_trial_pairs(n_trials):
    pair_idx = np.array(list(combinations(np.arange(n_trials),2)))
    pair_idx = np.vstack((pair_idx, np.fliplr(pair_idx))) # Ensures covXX = covYY
    return pair_idx

if __name__ == "__main__":
    from CovarianceEstimators import MaximumLikelihoodCovarianceEstimator

    fname = "/home/groups/amnorcia/eeg_data/kaneshiro_data/kaneshiro_eeg.npy"
    d = np.load(fname)
    d = np.transpose(d, (0,1,2,4,3))
    print d.shape

    c = MaximumLikelihoodCovarianceEstimator("mle")

    import time
    start = time.time()

    avgCovXX, avgCovXY, avgCovYY = compute_rca_covariance_subj_condition(d, c)

    print "Time elapsed: {} s".format(time.time() - start)

    np.save("covariance_matrices/avgCovXX.npy", avgCovXX)
    np.save("covariance_matrices/avgCovXY.npy", avgCovXY)
    np.save("covariance_matrices/avgCovYY.npy", avgCovYY)


